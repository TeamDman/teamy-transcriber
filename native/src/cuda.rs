use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::rc::Rc;

unsafe extern "C" {
    fn tw_error() -> *const c_char;
    fn tw_create(out: *mut *mut c_void, device: i32, tf32: i32) -> i32;
    fn tw_destroy(s: *mut c_void);
    fn tw_alloc(s: *mut c_void, out: *mut *mut f32, n: usize) -> i32;
    fn tw_free(s: *mut c_void, p: *mut f32);
    fn tw_upload(s: *mut c_void, dst: *mut f32, src: *const f32, n: usize) -> i32;
    fn tw_download(s: *mut c_void, dst: *mut f32, src: *const f32, n: usize) -> i32;
    fn tw_copy(s: *mut c_void, dst: *mut f32, src: *const f32, n: usize) -> i32;
    fn tw_sync(s: *mut c_void) -> i32;
    fn tw_linear(
        s: *mut c_void,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        r: *const f32,
        y: *mut f32,
        rows: i32,
        input: i32,
        output: i32,
        gelu: i32,
    ) -> i32;
    fn tw_norm(
        s: *mut c_void,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        y: *mut f32,
        rows: i32,
        width: i32,
    ) -> i32;
    fn tw_conv(
        s: *mut c_void,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        col: *mut f32,
        y: *mut f32,
        time: i32,
        input: i32,
        output: i32,
        stride: i32,
        planar: i32,
    ) -> i32;
    fn tw_position(s: *mut c_void, x: *mut f32, pos: *const f32, n: i32) -> i32;
    fn tw_embed(
        s: *mut c_void,
        w: *const f32,
        pos: *const f32,
        y: *mut f32,
        token: i32,
        position: i32,
        width: i32,
    ) -> i32;
    fn tw_attention(
        s: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        scores: *mut f32,
        out: *mut f32,
        nq: i32,
        nk: i32,
        width: i32,
        heads: i32,
        causal: i32,
        offset: i32,
    ) -> i32;
    fn tw_argmax(
        s: *mut c_void,
        x: *const f32,
        allowed: *const f32,
        result: *mut f32,
        n: i32,
    ) -> i32;
    fn tw_decode_attention(
        s: *mut c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        batch: i32,
        keys: i32,
        capacity: i32,
        width: i32,
        heads: i32,
    ) -> i32;
    fn tw_cache_token(
        s: *mut c_void,
        x: *const f32,
        cache: *mut f32,
        batch: i32,
        position: i32,
        capacity: i32,
        width: i32,
    ) -> i32;
    fn tw_embed_batch(
        s: *mut c_void,
        w: *const f32,
        pos: *const f32,
        tokens: *const f32,
        y: *mut f32,
        batch: i32,
        position: i32,
        width: i32,
    ) -> i32;
    fn tw_argmax_batch(
        s: *mut c_void,
        x: *const f32,
        allowed: *const f32,
        result: *mut f32,
        batch: i32,
        n: i32,
    ) -> i32;
}

fn checked(code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        // SAFETY: the shim returns a NUL-terminated thread-local error buffer.
        Err(anyhow!(
            unsafe { CStr::from_ptr(tw_error()) }
                .to_string_lossy()
                .into_owned()
        ))
    }
}

#[derive(Debug)]
pub(crate) struct Device(NonNull<c_void>);
impl Device {
    pub fn new(index: i32, tf32: bool) -> Result<Rc<Self>> {
        ensure!(index >= 0, "CUDA device must be nonnegative");
        let mut raw = std::ptr::null_mut();
        // SAFETY: out is valid, and the shim initializes it only on success.
        checked(unsafe { tw_create(&mut raw, index, i32::from(tf32)) })?;
        Ok(Rc::new(Self(
            NonNull::new(raw).ok_or_else(|| anyhow!("null CUDA session"))?,
        )))
    }
    pub fn sync(&self) -> Result<()> {
        // SAFETY: this Rc-owned session stays live through the call.
        checked(unsafe { tw_sync(self.0.as_ptr()) })
    }
    pub fn alloc(self: &Rc<Self>, n: usize) -> Result<Buffer> {
        ensure!(
            n > 0 && n <= i32::MAX as usize,
            "invalid CUDA buffer size {n}"
        );
        let mut raw = std::ptr::null_mut();
        // SAFETY: checked size fits bytes; the allocation is exclusively owned below.
        checked(unsafe { tw_alloc(self.0.as_ptr(), &mut raw, n) })?;
        let raw = NonNull::new(raw).ok_or_else(|| anyhow!("null CUDA buffer"))?;
        Ok(Buffer {
            raw,
            n,
            allocation: Rc::new(Allocation {
                raw,
                device: Rc::clone(self),
            }),
        })
    }
    pub fn upload(self: &Rc<Self>, values: &[f32]) -> Result<Buffer> {
        let out = self.alloc(values.len())?;
        out.write(values)?;
        Ok(out)
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: all buffers hold an Rc; this is the last session owner.
        unsafe { tw_destroy(self.0.as_ptr()) };
    }
}

/// The base allocation owns the only free; checked subranges keep it alive.
#[derive(Debug)]
struct Allocation {
    raw: NonNull<f32>,
    device: Rc<Device>,
}
impl Drop for Allocation {
    fn drop(&mut self) {
        // SAFETY: all subranges are gone; the live session synchronizes before free.
        unsafe { tw_free(self.device.0.as_ptr(), self.raw.as_ptr()) };
    }
}

/// Thread-confined GPU range. Rc deliberately makes the engine !Send/!Sync.
#[derive(Debug)]
pub(crate) struct Buffer {
    raw: NonNull<f32>,
    pub n: usize,
    allocation: Rc<Allocation>,
}
impl Buffer {
    fn ptr(&self) -> *mut f32 {
        self.raw.as_ptr()
    }
    fn session(&self) -> *mut c_void {
        self.allocation.device.0.as_ptr()
    }
    fn fits(&self, n: usize) -> Result<()> {
        ensure!(n <= self.n, "CUDA buffer bounds: {n} > {}", self.n);
        Ok(())
    }
    fn same(&self, buffers: &[&Self]) -> Result<()> {
        ensure!(
            buffers
                .iter()
                .all(|b| Rc::ptr_eq(&self.allocation.device, &b.allocation.device)),
            "CUDA buffers belong to different sessions"
        );
        Ok(())
    }
    pub fn slice(&self, offset: usize, n: usize) -> Result<Self> {
        ensure!(n > 0, "empty CUDA subrange");
        self.fits(
            offset
                .checked_add(n)
                .ok_or_else(|| anyhow!("offset overflow"))?,
        )?;
        Ok(Self {
            // The address calculation does not dereference device memory.
            raw: NonNull::new(self.ptr().wrapping_add(offset)).unwrap(),
            n,
            allocation: Rc::clone(&self.allocation),
        })
    }
    pub fn write(&self, values: &[f32]) -> Result<()> {
        self.fits(values.len())?;
        // SAFETY: lengths checked; upload synchronizes before host memory is released.
        checked(unsafe { tw_upload(self.session(), self.ptr(), values.as_ptr(), values.len()) })
    }
    pub fn read(&self, n: usize) -> Result<Vec<f32>> {
        self.fits(n)?;
        let mut values = vec![0.; n];
        // SAFETY: both slices have n elements; download synchronizes its stream.
        checked(unsafe { tw_download(self.session(), values.as_mut_ptr(), self.ptr(), n) })?;
        Ok(values)
    }
    pub fn copy_from(&self, source: &Self, offset: usize, n: usize) -> Result<()> {
        self.same(&[source])?;
        source.fits(n)?;
        self.fits(
            offset
                .checked_add(n)
                .ok_or_else(|| anyhow!("offset overflow"))?,
        )?;
        // SAFETY: source and destination bounds checked; owned stream orders reads/writes.
        checked(unsafe { tw_copy(self.session(), self.ptr().add(offset), source.ptr(), n) })
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit dimensions and operands mirror the checked CUDA matrix operation."
    )]
    pub fn linear(
        &self,
        w: &Self,
        b: Option<&Self>,
        residual: Option<&Self>,
        y: &Self,
        rows: usize,
        input: usize,
        output: usize,
        gelu: bool,
    ) -> Result<()> {
        self.same(&[w, y])?;
        self.fits(rows * input)?;
        w.fits(input * output)?;
        y.fits(rows * output)?;
        if let Some(b) = b {
            self.same(&[b])?;
            b.fits(output)?;
        }
        if let Some(r) = residual {
            self.same(&[r])?;
            r.fits(rows * output)?;
        }
        // SAFETY: all dimensions/buffers validated above; input/output are separately allocated by graph.
        checked(unsafe {
            tw_linear(
                self.session(),
                self.ptr(),
                w.ptr(),
                b.map_or(std::ptr::null(), |b| b.ptr()),
                residual.map_or(std::ptr::null(), |b| b.ptr()),
                y.ptr(),
                rows as i32,
                input as i32,
                output as i32,
                i32::from(gelu),
            )
        })
    }
    pub fn norm(&self, w: &Self, b: &Self, y: &Self, rows: usize, width: usize) -> Result<()> {
        self.same(&[w, b, y])?;
        self.fits(rows * width)?;
        y.fits(rows * width)?;
        w.fits(width)?;
        b.fits(width)?;
        // SAFETY: row and affine buffers validated; reduction supports non-power-of-two widths.
        checked(unsafe {
            tw_norm(
                self.session(),
                self.ptr(),
                w.ptr(),
                b.ptr(),
                y.ptr(),
                rows as i32,
                width as i32,
            )
        })
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit convolution dimensions and layout mirror the checked CUDA operation."
    )]
    pub fn conv(
        &self,
        w: &Self,
        b: &Self,
        col: &Self,
        y: &Self,
        time: usize,
        input: usize,
        output: usize,
        stride: usize,
        planar: bool,
    ) -> Result<()> {
        ensure!(stride == 1 || stride == 2, "invalid convolution stride");
        let rows = time.div_ceil(stride);
        self.same(&[w, b, col, y])?;
        self.fits(time * input)?;
        w.fits(input * output * 3)?;
        b.fits(output)?;
        col.fits(rows * input * 3)?;
        y.fits(rows * output)?;
        // SAFETY: input, weight, im2col and output extents checked.
        checked(unsafe {
            tw_conv(
                self.session(),
                self.ptr(),
                w.ptr(),
                b.ptr(),
                col.ptr(),
                y.ptr(),
                time as i32,
                input as i32,
                output as i32,
                stride as i32,
                i32::from(planar),
            )
        })
    }
    pub fn position(&self, pos: &Self, n: usize) -> Result<()> {
        self.same(&[pos])?;
        self.fits(n)?;
        pos.fits(n)?;
        // SAFETY: elementwise in-place add of checked extents.
        checked(unsafe { tw_position(self.session(), self.ptr(), pos.ptr(), n as i32) })
    }
    pub fn embedding(
        &self,
        pos: &Self,
        y: &Self,
        token: usize,
        position: usize,
        width: usize,
    ) -> Result<()> {
        self.same(&[pos, y])?;
        self.fits((token + 1) * width)?;
        pos.fits((position + 1) * width)?;
        y.fits(width)?;
        // SAFETY: selected rows checked before launching the gather.
        checked(unsafe {
            tw_embed(
                self.session(),
                self.ptr(),
                pos.ptr(),
                y.ptr(),
                token as i32,
                position as i32,
                width as i32,
            )
        })
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Query/key extents, heads and causal offset are checked at the CUDA boundary."
    )]
    pub fn attention(
        &self,
        k: &Self,
        v: &Self,
        scores: &Self,
        out: &Self,
        nq: usize,
        nk: usize,
        width: usize,
        heads: usize,
        causal: bool,
        offset: usize,
    ) -> Result<()> {
        ensure!(
            heads > 0 && width.is_multiple_of(heads) && nq > 0 && nk > 0,
            "invalid attention shape"
        );
        self.same(&[k, v, scores, out])?;
        self.fits(nq * width)?;
        k.fits(nk * width)?;
        v.fits(nk * width)?;
        scores.fits(heads * nq * nk)?;
        out.fits(nq * width)?;
        // SAFETY: strided cuBLAS batches remain inside checked interleaved head buffers.
        checked(unsafe {
            tw_attention(
                self.session(),
                self.ptr(),
                k.ptr(),
                v.ptr(),
                scores.ptr(),
                out.ptr(),
                nq as i32,
                nk as i32,
                width as i32,
                heads as i32,
                i32::from(causal),
                offset as i32,
            )
        })
    }
    pub fn argmax(&self, allowed: &Self, result: &Self, n: usize) -> Result<usize> {
        self.same(&[allowed, result])?;
        self.fits(n)?;
        allowed.fits(n)?;
        result.fits(1)?;
        // SAFETY: checked vocabulary mask and one-element result.
        checked(unsafe {
            tw_argmax(
                self.session(),
                self.ptr(),
                allowed.ptr(),
                result.ptr(),
                n as i32,
            )
        })?;
        let id = result.read(1)?[0];
        ensure!(id >= 0. && id < (n as f32), "no finite unsuppressed logits");
        Ok(id as usize)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit batch/cache dimensions are checked before CUDA launch."
    )]
    pub fn decode_attention(
        &self,
        k: &Self,
        v: &Self,
        out: &Self,
        batch: usize,
        keys: usize,
        capacity: usize,
        width: usize,
        heads: usize,
    ) -> Result<()> {
        ensure!(
            (1..=8).contains(&batch)
                && (1..=1500).contains(&keys)
                && (keys..=1500).contains(&capacity)
                && (1..=20).contains(&heads)
                && width == heads * 64,
            "invalid batched decoder attention shape"
        );
        self.same(&[k, v, out])?;
        self.fits(batch * width)?;
        k.fits(batch * capacity * width)?;
        v.fits(batch * capacity * width)?;
        out.fits(batch * width)?;
        // SAFETY: fixed 64-wide heads, bounded shared scores, all batch/cache extents checked.
        checked(unsafe {
            tw_decode_attention(
                self.session(),
                self.ptr(),
                k.ptr(),
                v.ptr(),
                out.ptr(),
                batch as i32,
                keys as i32,
                capacity as i32,
                width as i32,
                heads as i32,
            )
        })
    }

    pub fn cache_token(
        &self,
        cache: &Self,
        batch: usize,
        position: usize,
        capacity: usize,
        width: usize,
    ) -> Result<()> {
        ensure!(
            (1..=8).contains(&batch)
                && (1..=448).contains(&capacity)
                && position < capacity
                && (1..=1280).contains(&width),
            "invalid token cache shape"
        );
        self.same(&[cache])?;
        self.fits(batch * width)?;
        cache.fits(batch * capacity * width)?;
        // SAFETY: one destination row per independent sequence is in bounds.
        checked(unsafe {
            tw_cache_token(
                self.session(),
                self.ptr(),
                cache.ptr(),
                batch as i32,
                position as i32,
                capacity as i32,
                width as i32,
            )
        })
    }

    pub fn argmax_batch(
        &self,
        allowed: &Self,
        result: &Self,
        batch: usize,
        n: usize,
    ) -> Result<Vec<usize>> {
        ensure!(
            (1..=8).contains(&batch) && (1..=60000).contains(&n),
            "invalid batched vocabulary"
        );
        self.same(&[allowed, result])?;
        self.fits(batch * n)?;
        allowed.fits(n)?;
        result.fits(batch)?;
        // SAFETY: every reduction reads one checked vocabulary row and writes one result.
        checked(unsafe {
            tw_argmax_batch(
                self.session(),
                self.ptr(),
                allowed.ptr(),
                result.ptr(),
                batch as i32,
                n as i32,
            )
        })?;
        let values = result.read(batch)?;
        ensure!(
            values.iter().all(|&id| id >= 0. && id < n as f32),
            "no finite unsuppressed batch logits"
        );
        Ok(values.into_iter().map(|id| id as usize).collect())
    }

    /// Device-resident token gather avoids a second host transfer each step.
    ///
    /// # Safety
    /// Each token must be an exact nonnegative integer indexing this embedding
    /// table (`token < self.n / width`). Writes establishing these values must
    /// precede this call on the same stream; no other writer may replace them.
    pub unsafe fn embedding_batch(
        &self,
        pos: &Self,
        tokens: &Self,
        y: &Self,
        batch: usize,
        position: usize,
        width: usize,
    ) -> Result<()> {
        ensure!(
            (1..=8).contains(&batch) && position < 448 && (1..=1280).contains(&width),
            "invalid batched embedding shape"
        );
        self.same(&[pos, tokens, y])?;
        pos.fits((position + 1) * width)?;
        tokens.fits(batch)?;
        y.fits(batch * width)?;
        // SAFETY: caller guarantees valid token indices into this embedding table;
        // position/output/token storage extents and session checked above.
        checked(unsafe {
            tw_embed_batch(
                self.session(),
                self.ptr(),
                pos.ptr(),
                tokens.ptr(),
                y.ptr(),
                batch as i32,
                position as i32,
                width as i32,
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!((a - b).abs() < tolerance, "index {i}: {a} != {b}");
        }
    }
    #[test]
    #[ignore = "requires a CUDA device; run explicitly in release mode"]
    fn batched_decoder_primitives_keep_sequences_independent() -> Result<()> {
        let device = Device::new(0, false)?;
        for (batch, heads, keys, capacity) in [
            (1, 1, 1, 8),
            (3, 2, 7, 11),
            (8, 20, 448, 448),
            (3, 20, 1500, 1500),
        ] {
            let width = heads * 64;
            let qs: Vec<f32> = (0..batch * width)
                .map(|i| (i as f32 * 0.017).sin())
                .collect();
            let mut ks = vec![1000.; batch * capacity * width];
            let mut vs = ks.clone();
            for b in 0..batch {
                for k in 0..keys {
                    for c in 0..width {
                        let index = (b * capacity + k) * width + c;
                        ks[index] = ((index + 13) as f32 * 0.029).cos();
                        vs[index] = ((index + 31) as f32 * 0.007).sin();
                    }
                }
            }
            let q = device.upload(&qs)?;
            let k = device.upload(&ks)?;
            let v = device.upload(&vs)?;
            let output = device.alloc(batch * width)?;
            q.decode_attention(&k, &v, &output, batch, keys, capacity, width, heads)?;
            let mut expected = vec![0.; batch * width];
            for b in 0..batch {
                for h in 0..heads {
                    let qb = b * width + h * 64;
                    let kb = b * capacity * width + h * 64;
                    let scores: Vec<f64> = (0..keys)
                        .map(|j| {
                            (0..64)
                                .map(|c| f64::from(qs[qb + c]) * f64::from(ks[kb + j * width + c]))
                                .sum::<f64>()
                                * 0.125
                        })
                        .collect();
                    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let probabilities: Vec<_> =
                        scores.iter().map(|s| (s - maximum).exp()).collect();
                    let denominator: f64 = probabilities.iter().sum();
                    for c in 0..64 {
                        expected[qb + c] = (probabilities
                            .iter()
                            .enumerate()
                            .map(|(j, p)| p / denominator * f64::from(vs[kb + j * width + c]))
                            .sum::<f64>()) as f32;
                    }
                }
            }
            close(&output.read(batch * width)?, &expected, 2e-5);
            assert!(
                q.decode_attention(&k, &v, &output, batch, capacity + 1, capacity, width, heads)
                    .is_err()
            );
        }
        let batch = 3;
        let width = 64;
        let capacity = 9;
        let input: Vec<_> = (0..batch * width).map(|i| i as f32 * 0.01).collect();
        let x = device.upload(&input)?;
        let cache = device.upload(&vec![-123.; batch * capacity * width])?;
        x.cache_token(&cache, batch, 4, capacity, width)?;
        let actual = cache.read(batch * capacity * width)?;
        for b in 0..batch {
            for p in 0..capacity {
                for c in 0..width {
                    assert_eq!(
                        actual[(b * capacity + p) * width + c],
                        if p == 4 { input[b * width + c] } else { -123. }
                    );
                }
            }
        }
        assert!(
            x.cache_token(&cache, batch, capacity, capacity, width)
                .is_err()
        );
        let logits = device.upload(&[0., 2., 1., 3., 2., 1., 0., 4., 3.])?;
        let mask = device.upload(&[1., 1., 0.])?;
        let result = device.alloc(batch)?;
        assert_eq!(logits.argmax_batch(&mask, &result, batch, 3)?, [1, 0, 1]);
        let embedding: Vec<_> = (0..3 * width).map(|i| i as f32 * 0.01).collect();
        let position = device.upload(&vec![0.5; width * 2])?;
        let table = device.upload(&embedding)?;
        // SAFETY: the checked argmax above only produces indices in this table.
        unsafe {
            table.embedding_batch(&position, &result, &x, batch, 1, width)?;
        }
        let expected: Vec<_> = [1, 0, 1]
            .into_iter()
            .flat_map(|id| {
                embedding[id * width..(id + 1) * width]
                    .iter()
                    .map(|v| v + 0.5)
            })
            .collect();
        close(&x.read(batch * width)?, &expected, 1e-6);
        assert!(
            logits
                .argmax_batch(&device.upload(&[0.; 3])?, &result, batch, 3)
                .is_err()
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires a CUDA device; run explicitly in release mode"]
    fn single_row_linear_matches_scalar_oracle() -> Result<()> {
        let device = Device::new(0, false)?;
        // Covers scalar tails, vectorized reads and deliberately unaligned
        // suballocations, with the same shapes used by large-v3's decoder.
        for input in [3, 37, 1280, 5120] {
            let output = 7;
            let xs: Vec<f32> = (0..input).map(|i| (i % 19) as f32 * 0.03 - 0.27).collect();
            let ws: Vec<f32> = (0..input * output)
                .map(|i| (i % 23) as f32 * 0.02 - 0.22)
                .collect();
            let bs: Vec<f32> = (0..output).map(|i| i as f32 * 0.05 - 0.1).collect();
            let rs: Vec<f32> = (0..output).map(|i| i as f32 * 0.07 - 0.2).collect();
            for offset in [0, 1] {
                let mut padded_x = vec![0.; offset];
                padded_x.extend_from_slice(&xs);
                let mut padded_w = vec![0.; offset];
                padded_w.extend_from_slice(&ws);
                let x = device.upload(&padded_x)?.slice(offset, input)?;
                let w = device.upload(&padded_w)?.slice(offset, input * output)?;
                let b = device.upload(&bs)?;
                let y = device.alloc(output)?;
                for fused in [false, true] {
                    y.write(&rs)?;
                    x.linear(
                        &w,
                        fused.then_some(&b),
                        fused.then_some(&y),
                        &y,
                        1,
                        input,
                        output,
                        fused,
                    )?;
                    let expected: Vec<f32> = (0..output)
                        .map(|row| {
                            let mut value: f64 = (0..input)
                                .map(|j| f64::from(xs[j]) * f64::from(ws[row * input + j]))
                                .sum();
                            if fused {
                                value += f64::from(bs[row]);
                                let z = value / std::f64::consts::SQRT_2;
                                let t = 1. / (1. + 0.3275911 * z.abs());
                                let erf = z.signum()
                                    * (1.
                                        - (((((1.061405429 * t - 1.453152027) * t)
                                            + 1.421413741)
                                            * t
                                            - 0.284496736)
                                            * t
                                            + 0.254829592)
                                            * t
                                            * (-z * z).exp());
                                value = 0.5 * value * (1. + erf) + f64::from(rs[row]);
                            }
                            value as f32
                        })
                        .collect();
                    close(&y.read(output)?, &expected, 5e-5);
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires a CUDA device; run explicitly in release mode"]
    fn cuda_primitives_match_scalar_oracles() -> Result<()> {
        let d = Device::new(0, false)?;
        let input = vec![1., 2., 3., 4., 5., 6.];
        let x = d.upload(&input)?;
        let arena = d.upload(&[10., 20., 30., 40.])?;
        let middle = arena.slice(1, 2)?;
        assert!(arena.slice(3, 2).is_err());
        assert!(middle.slice(1, 2).is_err());
        drop(arena);
        close(&middle.read(2)?, &[20., 30.], 1e-6);
        let w = d.upload(&[1., 0., -1., 0.5, 0.25, 0.125])?;
        let bias = d.upload(&[0.5, -0.5])?;
        let y = d.alloc(4)?;
        x.linear(&w, Some(&bias), None, &y, 2, 3, 2, false)?;
        close(&y.read(4)?, &[-1.5, 0.875, -1.5, 3.5], 1e-6);
        let scale = d.upload(&[1., 1., 1.])?;
        let shift = d.upload(&[0., 0., 0.])?;
        let normed = d.alloc(6)?;
        // Width 3 deliberately gives some reduction threads no input lanes.
        for _ in 0..8 {
            x.norm(&scale, &shift, &normed, 2, 3)?;
            close(
                &normed.read(6)?,
                &[-1.2247356, 0., 1.2247356, -1.2247356, 0., 1.2247356],
                2e-5,
            );
        }
        let n = 128;
        let nq = 3;
        let nk = 5;
        let heads = 2;
        let width = n / heads;
        let qs: Vec<_> = (0..nq * n).map(|i| (i % 17) as f32 * 0.01 - 0.08).collect();
        let ks: Vec<_> = (0..nk * n).map(|i| (i % 13) as f32 * 0.02 - 0.12).collect();
        let vs: Vec<_> = (0..nk * n).map(|i| (i % 19) as f32 * 0.03 - 0.27).collect();
        let q = d.upload(&qs)?;
        let k = d.upload(&ks)?;
        let v = d.upload(&vs)?;
        let scores = d.alloc(nq * nk * heads)?;
        let out = d.alloc(nq * n)?;
        q.attention(&k, &v, &scores, &out, nq, nk, n, heads, true, 1)?;
        let mut expected = vec![0.; nq * n];
        for row in 0..nq {
            for head in 0..heads {
                let count = (row + 2).min(nk);
                let mut probability: Vec<f32> = (0..count)
                    .map(|col| {
                        (0..width)
                            .map(|j| {
                                qs[row * n + head * width + j] * ks[col * n + head * width + j]
                            })
                            .sum::<f32>()
                            / (width as f32).sqrt()
                    })
                    .collect();
                let max = probability
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max);
                for p in &mut probability {
                    *p = (*p - max).exp();
                }
                let sum: f32 = probability.iter().sum();
                for j in 0..width {
                    expected[row * n + head * width + j] = (0..count)
                        .map(|col| probability[col] / sum * vs[col * n + head * width + j])
                        .sum();
                }
            }
        }
        close(&out.read(nq * n)?, &expected, 1e-5);
        // Convolution must agree at both padded boundaries, for the planar
        // spectrogram and the row-major second convolution, including odd time.
        for planar in [false, true] {
            for stride in [1, 2] {
                let time: usize = 5;
                let channels = 2;
                let outputs = 3;
                let rows = time.div_ceil(stride);
                let input: Vec<f32> = (0..time * channels)
                    .map(|i| (i as f32 - 4.) * 0.125)
                    .collect();
                let weights: Vec<f32> = (0..outputs * channels * 3)
                    .map(|i| (i as f32 - 8.) * 0.0625)
                    .collect();
                let biases = [0.25, -0.125, 0.5];
                let x = d.upload(&input)?;
                let w = d.upload(&weights)?;
                let b = d.upload(&biases)?;
                let col = d.alloc(rows * channels * 3)?;
                let y = d.alloc(rows * outputs)?;
                x.conv(&w, &b, &col, &y, time, channels, outputs, stride, planar)?;
                let mut expected = Vec::new();
                for row in 0..rows {
                    for output in 0..outputs {
                        let mut value = biases[output];
                        for channel in 0..channels {
                            for tap in 0..3 {
                                let t = (row * stride + tap) as isize - 1;
                                if (0..time as isize).contains(&t) {
                                    let i = if planar {
                                        channel * time + t as usize
                                    } else {
                                        t as usize * channels + channel
                                    };
                                    value +=
                                        input[i] * weights[(output * channels + channel) * 3 + tap];
                                }
                            }
                        }
                        // Normal-CDF GELU via an independent erf approximation
                        // (Abramowitz/Stegun), accurate enough for this oracle.
                        let z = f64::from(value) / std::f64::consts::SQRT_2;
                        let t = 1. / (1. + 0.3275911 * z.abs());
                        let erf = z.signum()
                            * (1.
                                - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                                    - 0.284496736)
                                    * t
                                    + 0.254829592)
                                    * t
                                    * (-z * z).exp());
                        expected.push((0.5 * f64::from(value) * (1. + erf)) as f32);
                    }
                }
                close(&y.read(expected.len())?, &expected, 2e-6);
            }
        }
        let logits = d.upload(&[1., 5., 5., 9.])?;
        let allowed = d.upload(&[1., 1., 1., 0.])?;
        let result = d.alloc(1)?;
        assert_eq!(logits.argmax(&allowed, &result, 4)?, 2);
        assert!(y.copy_from(&x, 0, 6).is_err());
        Ok(())
    }
}
