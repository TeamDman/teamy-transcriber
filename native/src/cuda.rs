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
            heads > 0 && width % heads == 0 && nq > 0 && nk > 0,
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
