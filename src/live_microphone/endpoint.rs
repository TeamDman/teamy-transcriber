//! Streaming Silero endpointing. Preserve native-rate PCM; resample only detector input.
use eyre::Context;
use eyre::Result;
use std::path::Path;
use teamy_whisper_native::vad::Silero;
use teamy_whisper_native::vad::WINDOW;

pub(super) struct Detector {
    model: Silero,
    window: [f32; WINDOW],
    used: usize,
    phase: u32,
    sum: f64,
    endpoint: Endpoint,
}

#[derive(Default)]
struct Endpoint {
    speech_frames: usize,
    silence_frames: usize,
}

impl Endpoint {
    fn step(&mut self, probability: f32) -> bool {
        if probability >= 0.5 {
            self.speech_frames += 1;
            self.silence_frames = 0;
        } else if probability < 0.35 && self.speech_frames > 0 {
            self.silence_frames += 1;
        } else {
            self.silence_frames = 0;
        }
        // 96ms of confirmed speech followed by 320ms of quiet.
        self.speech_frames >= 3 && self.silence_frames >= 10
    }
}

impl Detector {
    pub(super) fn load(model_dir: &Path) -> Result<Option<Self>> {
        let path = model_dir.join(crate::native_whisper::speech::DIRECTORY);
        if !path.exists() {
            eprintln!("Selected model has no VAD; using fixed maximum windows.");
            return Ok(None);
        }
        let (model, _) =
            crate::native_whisper::speech::load(&path).wrap_err("loading live microphone VAD")?;
        Ok(Some(Self {
            model,
            window: [0.; WINDOW],
            used: 0,
            phase: 0,
            sum: 0.,
            endpoint: Endpoint::default(),
        }))
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded weighted PCM average fits f32"
    )]
    pub(super) fn push(&mut self, sample: f32, rate: u32) -> Result<bool> {
        // Area-weighted resampling with integer phase preserves timing across
        // arbitrary callback boundaries, including 44.1kHz devices.
        let mut remaining = 16000;
        let mut submit = false;
        while remaining > 0 {
            let count = remaining.min(rate - self.phase);
            self.sum += f64::from(sample.clamp(-1., 1.)) * f64::from(count);
            self.phase += count;
            remaining -= count;
            if self.phase == rate {
                self.window[self.used] = (self.sum / f64::from(rate)) as f32;
                self.used += 1;
                self.phase = 0;
                self.sum = 0.;
                if self.used == WINDOW {
                    let probability = self
                        .model
                        .step(&self.window)
                        .map_err(|error| eyre::eyre!(error.to_string()))?;
                    submit |= self.endpoint.step(probability);
                    self.used = 0;
                }
            }
        }
        Ok(submit)
    }

    pub(super) fn boundary(&mut self) {
        self.endpoint = Endpoint::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn submits_only_after_speech_and_a_sustained_pause() {
        let mut endpoint = Endpoint::default();
        for _ in 0..100 {
            assert!(!endpoint.step(0.));
        }
        for _ in 0..3 {
            assert!(!endpoint.step(0.9));
        }
        for _ in 0..9 {
            assert!(!endpoint.step(0.1));
        }
        assert!(!endpoint.step(0.9)); // brief pause must not split the utterance
        for _ in 0..9 {
            assert!(!endpoint.step(0.1));
        }
        assert!(endpoint.step(0.1));
    }
    #[test]
    #[ignore = "requires TEST_MODEL with prepared Silero weights"]
    fn detector_resampling_preserves_time_for_microphone_rates() -> Result<()> {
        let model = std::path::PathBuf::from(std::env::var("TEST_MODEL")?);
        for rate in [8000, 16000, 44100, 48000] {
            let mut detector = Detector::load(&model)?.expect("fixture must include VAD");
            for _ in 0..rate {
                assert!(!detector.push(0., rate)?);
            }
            assert_eq!(detector.used, 16000 % WINDOW);
            assert_eq!(detector.phase, 0);
            assert!(detector.sum.abs() < f64::EPSILON);
        }
        Ok(())
    }
}
