//! Bounded independent audio sequences sharing weights and decoder launches.
use super::*;

pub const MAX_BATCH_SIZE: usize = 8;

#[derive(Debug, Serialize)]
pub struct BatchTranscript {
    pub text: String,
    pub tokens: Vec<usize>,
    pub ended: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA, WHISPER_BATCH_TEST_MODEL and WHISPER_BATCH_TEST_WAV"]
    fn batch_reuse_growth_reordering_and_early_end_match_serial() -> Result<()> {
        let root = std::env::var("WHISPER_BATCH_TEST_MODEL")?;
        let path = std::env::var("WHISPER_BATCH_TEST_WAV")?;
        let mut wav = hound::WavReader::open(path)?;
        ensure!(
            wav.spec().sample_rate == 16000 && wav.spec().channels == 1,
            "expected 16 kHz mono"
        );
        let audio: Vec<f32> = match wav.spec().sample_format {
            hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<_, _>>()?,
            hound::SampleFormat::Int => wav
                .samples::<i16>()
                .map(|s| s.map(|s| f32::from(s) / 32768.))
                .collect::<Result<_, _>>()?,
        };
        let mut engine = Engine::load(Path::new(&root), 0, false)?;
        let mut frontend = frontend::Frontend::new(engine.dims.audio.n_mels)?;
        let mels = [
            frontend.compute(&audio)?,
            frontend.compute(&vec![0.; 16000])?,
            frontend.compute(&audio[..audio.len().min(8000)])?,
        ];
        let references = mels
            .iter()
            .map(|m| engine.transcribe_mel(m, 448))
            .collect::<Result<Vec<_>>>()?;
        assert!(
            references.iter().all(|r| r.ended),
            "fixture must terminate before token cap"
        );
        assert!(
            references.iter().map(|r| r.tokens.len()).min()
                != references.iter().map(|r| r.tokens.len()).max(),
            "fixture must exercise unequal output lengths"
        );
        assert_eq!(engine.batch_workspace_bytes(), 0);
        let mut previous_bytes = 0;
        for order in [
            vec![0, 1],
            vec![2, 0, 1, 2, 0],
            vec![1],
            vec![1, 0, 2],
            vec![2, 0, 1, 2, 0],
        ] {
            let inputs: Vec<_> = order.iter().map(|&i| mels[i].as_slice()).collect();
            let result = engine.transcribe_mel_batch(&inputs, 448)?;
            for (actual, &index) in result.transcripts.iter().zip(&order) {
                assert_eq!(actual.tokens, references[index].tokens, "slot {index}");
                assert_eq!(actual.ended, references[index].ended);
            }
            assert!(engine.batch_workspace_bytes() >= previous_bytes);
            previous_bytes = engine.batch_workspace_bytes();
        }
        assert!(engine.transcribe_mel_batch(&[], 448).is_err());
        assert!(
            engine
                .transcribe_mel_batch(&[mels[0].as_slice(); 9], 448)
                .is_err()
        );
        assert!(engine.transcribe_mel_batch(&[&mels[0]], 0).is_err());
        assert!(engine.transcribe_mel_batch(&[&[f32::NAN]], 448).is_err());
        assert_eq!(previous_bytes, engine.batch_workspace_bytes());
        let one = engine.transcribe_mel_batch(&[&mels[0], &mels[1]], 1)?;
        for (i, r) in one.transcripts.iter().enumerate() {
            let serial = engine.transcribe_mel(&mels[i], 1)?;
            assert_eq!(r.tokens, serial.tokens);
            assert_eq!(r.ended, serial.ended);
        }
        assert_eq!(
            engine.transcribe_mel(&mels[0], 448)?.tokens,
            references[0].tokens
        );
        // Cancel before encoding, during the first decoder step, and after a
        // persisted prefix. None may poison a later use of the shared caches.
        use std::cell::Cell;
        let stop = Cell::new(false);
        let inputs = [&mels[0][..], &mels[1][..], &mels[2][..]];
        let mut calls = 0;
        let before =
            engine.transcribe_mel_batch_with(&inputs, 448, &mut || true, &mut |_, _| {
                calls += 1;
                Ok(())
            })?;
        assert!(before.cancelled && before.transcripts.is_empty());
        assert_eq!(calls, 0);
        let mut checks = 0;
        let during = engine.transcribe_mel_batch_with(
            &inputs,
            448,
            &mut || {
                checks += 1;
                checks == 6
            },
            &mut |_, _| {
                calls += 1;
                Ok(())
            },
        )?;
        assert!(during.cancelled && during.transcripts.is_empty());
        assert_eq!(calls, 0);
        let prefix = engine.transcribe_mel_batch_with(
            &inputs,
            448,
            &mut || stop.get(),
            &mut |index, transcript| {
                assert_eq!(index, 0);
                assert_eq!(transcript.tokens, references[0].tokens);
                stop.set(true);
                Ok(())
            },
        )?;
        assert!(prefix.cancelled && prefix.transcripts.len() == 1);
        let mut checks = 0;
        assert!(
            engine
                .transcribe_mel_interruptible(&mels[0], 448, &mut || {
                    checks += 1;
                    checks == 3
                })?
                .is_none()
        );
        let again = engine.transcribe_mel_batch(&inputs, 448)?;
        assert!(!again.cancelled);
        assert_eq!(again.transcripts.len(), references.len());
        for (actual, expected) in again.transcripts.iter().zip(&references) {
            assert_eq!(actual.tokens, expected.tokens);
        }
        assert_eq!(
            engine.transcribe_mel(&mels[0], 448)?.tokens,
            references[0].tokens
        );
        assert!((1..=MAX_BATCH_SIZE).contains(&engine.fitting_batch_size(MAX_BATCH_SIZE)?));
        assert!(engine.fitting_batch_size(0).is_err());
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub struct BatchDecodeResult {
    pub transcripts: Vec<BatchTranscript>,
    /// Only the completed, delivered prefix is returned after cancellation.
    pub cancelled: bool,
    /// Serial encoders and prompt prefills for every input, with GPU completion.
    pub prepare_ms: f64,
    /// Concurrent greedy decoding, including host token selection and GPU completion.
    pub decode_ms: f64,
}

pub(super) struct BatchWorkspace {
    capacity: usize,
    allocated_bytes: usize,
    cache: Vec<Cache>,
    // Checked views into `cache`, not additional GPU allocations. These let
    // the established serial encoder/prompt path initialize independent slots.
    slots: Vec<Vec<Cache>>,
    x: Buffer,
    norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    context: Buffer,
    temp: Buffer,
    ff: Buffer,
    logits: Buffer,
    result: Buffer,
}
impl BatchWorkspace {
    fn bytes(dims: &Dims, batch: usize) -> usize {
        4 * batch
            * (2 * dims.text.n_text_layer
                * dims.text.n_text_state
                * (dims.text.n_text_ctx + dims.audio.n_audio_ctx)
                + 11 * dims.text.n_text_state
                + dims.text.n_vocab
                + 1)
    }
    fn new(device: &Rc<Device>, dims: &Dims, batch: usize) -> Result<Self> {
        ensure!(
            (1..=MAX_BATCH_SIZE).contains(&batch),
            "batch size must be 1..={MAX_BATCH_SIZE}"
        );
        let width = dims.text.n_text_state;
        let self_size = width * dims.text.n_text_ctx;
        let cross_size = width * dims.audio.n_audio_ctx;
        let cache: Vec<Cache> = (0..dims.text.n_text_layer)
            .map(|_| {
                Ok(Cache {
                    key: device.alloc(batch * self_size)?,
                    value: device.alloc(batch * self_size)?,
                    cross_key: device.alloc(batch * cross_size)?,
                    cross_value: device.alloc(batch * cross_size)?,
                })
            })
            .collect::<Result<_>>()?;
        let slots = (0..batch)
            .map(|slot| {
                cache
                    .iter()
                    .map(|c| {
                        Ok(Cache {
                            key: c.key.slice(slot * self_size, self_size)?,
                            value: c.value.slice(slot * self_size, self_size)?,
                            cross_key: c.cross_key.slice(slot * cross_size, cross_size)?,
                            cross_value: c.cross_value.slice(slot * cross_size, cross_size)?,
                        })
                    })
                    .collect::<Result<_>>()
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            capacity: batch,
            allocated_bytes: Self::bytes(dims, batch),
            cache,
            slots,
            x: device.alloc(batch * width)?,
            norm: device.alloc(batch * width)?,
            q: device.alloc(batch * width)?,
            k: device.alloc(batch * width)?,
            v: device.alloc(batch * width)?,
            context: device.alloc(batch * width)?,
            temp: device.alloc(batch * width)?,
            ff: device.alloc(batch * width * 4)?,
            logits: device.alloc(batch * dims.text.n_vocab)?,
            result: device.alloc(batch)?,
        })
    }
}

impl Engine {
    /// Leave 256 MiB free when sizing additional batch caches. A result of one
    /// selects the original serial path, which needs no batch allocation.
    pub fn fitting_batch_size(&self, requested: usize) -> Result<usize> {
        ensure!(
            (1..=MAX_BATCH_SIZE).contains(&requested),
            "invalid requested batch size"
        );
        let existing = self.batch.as_ref().map_or(0, |w| w.capacity);
        if existing >= requested {
            return Ok(requested);
        }
        let budget = self
            .device
            .free_bytes()?
            .saturating_add(self.batch_workspace_bytes())
            .saturating_sub(256 * 1024 * 1024);
        Ok(requested
            .min(budget / BatchWorkspace::bytes(&self.dims, 1))
            .max(1))
    }
    /// Extra allocated GPU memory for batch caches/scratch, excluding the shared
    /// weights and the original single-input workspace. Allocated lazily.
    pub fn batch_workspace_bytes(&self) -> usize {
        self.batch.as_ref().map_or(0, |w| w.allocated_bytes)
    }

    /// Decode independent windows, preserving input order and per-window EOT.
    /// Inputs share the configured prompt/suppression and token limit. Completed
    /// slots remain in the GPU batch but no more tokens are appended to them.
    /// Storage is reused and bounded to eight slots; allocation failure is an
    /// explicit error so callers can select a smaller batch. No weights are copied.
    pub fn transcribe_mel_batch(
        &mut self,
        mels: &[&[f32]],
        max_tokens: usize,
    ) -> Result<BatchDecodeResult> {
        self.transcribe_mel_batch_with(mels, max_tokens, &mut || false, &mut |_, _| Ok(()))
    }

    /// Deliver complete transcripts in input order. A completion callback may
    /// stop the request via `should_stop`; unfinished and undelivered slots are
    /// discarded on cancellation. Callbacks can persist results before returning.
    pub fn transcribe_mel_batch_with(
        &mut self,
        mels: &[&[f32]],
        max_tokens: usize,
        should_stop: &mut dyn FnMut() -> bool,
        on_complete: &mut dyn FnMut(usize, &BatchTranscript) -> Result<()>,
    ) -> Result<BatchDecodeResult> {
        let count = mels.len();
        ensure!(
            (1..=MAX_BATCH_SIZE).contains(&count),
            "batch size must be 1..={MAX_BATCH_SIZE}"
        );
        ensure!(max_tokens > 0, "max tokens must be positive");
        let mel_size = self.dims.audio.n_mels * self.dims.audio.n_audio_ctx * 2;
        ensure!(
            mels.iter()
                .all(|m| m.len() == mel_size && m.iter().all(|x| x.is_finite())),
            "expected finite Whisper mel matrices"
        );
        let started = Instant::now();
        if should_stop() {
            return Ok(BatchDecodeResult {
                transcripts: Vec::new(),
                cancelled: true,
                prepare_ms: 0.,
                decode_ms: 0.,
            });
        }
        if self.batch.as_ref().is_none_or(|w| w.capacity < count) {
            // Do not retain two large allocations while growing. A failed grow
            // releases partial allocations and leaves the serial engine usable.
            self.batch = None;
            self.batch = Some(BatchWorkspace::new(&self.device, &self.dims, count)?);
        }
        let w = self.batch.as_ref().unwrap();
        let width = self.dims.text.n_text_state;
        for (slot, mel) in mels.iter().enumerate() {
            if should_stop() {
                self.device.sync()?;
                return Ok(BatchDecodeResult {
                    transcripts: Vec::new(),
                    cancelled: true,
                    prepare_ms: started.elapsed().as_secs_f64() * 1000.,
                    decode_ms: 0.,
                });
            }
            self.encode_cached(mel, &w.slots[slot])?;
            self.decoder_tokens_cached(&self.prompt, 0, &w.slots[slot])?;
            w.norm.copy_from(
                &self
                    .work
                    .norm
                    .slice((self.prompt.len() - 1) * width, width)?,
                slot * width,
                width,
            )?;
        }
        self.device.sync()?;
        let prepare_ms = started.elapsed().as_secs_f64() * 1000.;
        let started = Instant::now();
        let mut tokens = vec![Vec::new(); count];
        let mut ended = vec![false; count];
        let mut transcripts = Vec::with_capacity(count);
        let mut cancelled = false;
        let limit = max_tokens.min(self.dims.text.n_text_ctx - self.prompt.len());
        for step in 0..limit {
            if should_stop() {
                cancelled = true;
                break;
            }
            w.norm.linear(
                self.output.as_ref().unwrap_or(&self.embed),
                None,
                None,
                &w.logits,
                count,
                width,
                self.dims.text.n_vocab,
                false,
            )?;
            let ids = w.logits.argmax_batch(
                if step == 0 {
                    self.begin_allowed.as_ref().unwrap_or(&self.allowed)
                } else {
                    &self.allowed
                },
                &w.result,
                count,
                self.dims.text.n_vocab,
            )?;
            for slot in 0..count {
                if !ended[slot] {
                    if ids[slot] == self.eot {
                        ended[slot] = true;
                    } else {
                        tokens[slot].push(ids[slot]);
                    }
                }
            }
            while transcripts.len() < count && (ended[transcripts.len()] || step + 1 == limit) {
                if should_stop() {
                    cancelled = true;
                    break;
                }
                let slot = transcripts.len();
                let tokens = std::mem::take(&mut tokens[slot]);
                let ids: Vec<_> = tokens.iter().map(|&id| id as u32).collect();
                let text = self
                    .tokenizer
                    .decode(&ids, true)
                    .map_err(|e| anyhow!(e.to_string()))?;
                let transcript = BatchTranscript {
                    text,
                    tokens,
                    ended: ended[slot],
                };
                on_complete(slot, &transcript)?;
                transcripts.push(transcript);
            }
            if cancelled {
                break;
            }
            if ended.iter().all(|&done| done) || step + 1 == limit {
                break;
            }
            self.decoder_batch_step(w, count, self.prompt.len() + step)?;
        }
        self.device.sync()?;
        let decode_ms = started.elapsed().as_secs_f64() * 1000.;
        Ok(BatchDecodeResult {
            transcripts,
            cancelled,
            prepare_ms,
            decode_ms,
        })
    }

    fn decoder_batch_step(&self, w: &BatchWorkspace, count: usize, position: usize) -> Result<()> {
        let t = &self.dims.text;
        let width = t.n_text_state;
        let audio = self.dims.audio.n_audio_ctx;
        // SAFETY: the preceding argmax_batch checked every result against the
        // tokenizer vocabulary, including finished slots. The result buffer has
        // not been modified; all buffers share the engine's ordered stream.
        unsafe {
            self.embed
                .embedding_batch(&self.text_pos, &w.result, &w.x, count, position, width)?;
        }
        for (layer, cache) in self.decoder.iter().zip(&w.cache) {
            layer.norm.run(&w.x, &w.norm, count, width)?;
            layer.attn.q.run(&w.norm, &w.q, count, None, false)?;
            layer.attn.k.run(&w.norm, &w.k, count, None, false)?;
            layer.attn.v.run(&w.norm, &w.v, count, None, false)?;
            w.k.cache_token(&cache.key, count, position, t.n_text_ctx, width)?;
            w.v.cache_token(&cache.value, count, position, t.n_text_ctx, width)?;
            w.q.decode_attention(
                &cache.key,
                &cache.value,
                &w.context,
                count,
                position + 1,
                t.n_text_ctx,
                width,
                t.n_text_head,
            )?;
            layer
                .attn
                .out
                .run(&w.context, &w.temp, count, Some(&w.x), false)?;
            let (norm, attn) = layer.cross.as_ref().unwrap();
            norm.run(&w.temp, &w.norm, count, width)?;
            attn.q.run(&w.norm, &w.q, count, None, false)?;
            w.q.decode_attention(
                &cache.cross_key,
                &cache.cross_value,
                &w.context,
                count,
                audio,
                audio,
                width,
                t.n_text_head,
            )?;
            attn.out
                .run(&w.context, &w.x, count, Some(&w.temp), false)?;
            layer.ff_norm.run(&w.x, &w.norm, count, width)?;
            layer.fc1.run(&w.norm, &w.ff, count, None, true)?;
            layer.fc2.run(&w.ff, &w.temp, count, Some(&w.x), false)?;
            w.x.copy_from(&w.temp, 0, count * width)?;
        }
        self.decoder_norm.run(&w.x, &w.norm, count, width)
    }
}
