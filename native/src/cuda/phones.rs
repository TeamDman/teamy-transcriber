use super::*;
unsafe extern "C" {
    fn px_conv(
        s: *mut c_void,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        col: *mut f32,
        y: *mut f32,
        time: i32,
        input: i32,
        output: i32,
        kernel: i32,
        stride: i32,
        pad: i32,
        groups: i32,
        rows: i32,
    ) -> i32;
    fn px_pointwise(
        s: *mut c_void,
        x: *const f32,
        b: *const f32,
        y: *mut f32,
        n: i32,
        op: i32,
        scale: f32,
    ) -> i32;
    fn px_rows(
        s: *mut c_void,
        x: *const f32,
        y: *mut f32,
        rows: i32,
        width: i32,
        xs: i32,
        ys: i32,
        xo: i32,
        yo: i32,
    ) -> i32;
    fn px_norm(
        s: *mut c_void,
        x: *const f32,
        w: *const f32,
        b: *const f32,
        y: *mut f32,
        rows: i32,
        width: i32,
        eps: f32,
    ) -> i32;
    fn px_softmax(s: *mut c_void, x: *const f32, y: *mut f32, rows: i32, width: i32) -> i32;
}
impl Buffer {
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit operands and dimensions mirror the bounds-checked CUDA convolution"
    )]
    pub(crate) fn phone_conv(
        &self,
        w: &Self,
        b: &Self,
        col: &Self,
        y: &Self,
        time: usize,
        input: usize,
        output: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        rows: usize,
    ) -> Result<()> {
        ensure!(
            groups > 0
                && stride > 0
                && input.is_multiple_of(groups)
                && output.is_multiple_of(groups)
                && kernel > 0,
            "invalid convolution"
        );
        ensure!(
            rows > 0 && (rows - 1) * stride + kernel <= time + 2 * pad,
            "invalid convolution length"
        );
        self.same(&[w, b, col, y])?;
        self.fits(time * input)?;
        w.fits(output * (input / groups) * kernel)?;
        b.fits(output)?;
        col.fits(rows * (input / groups) * kernel)?;
        y.fits(rows * output)?;
        // SAFETY: dimensions and all buffer bounds validated above.
        checked(unsafe {
            px_conv(
                self.session(),
                self.ptr(),
                w.ptr(),
                b.ptr(),
                col.ptr(),
                y.ptr(),
                time as i32,
                input as i32,
                output as i32,
                kernel as i32,
                stride as i32,
                pad as i32,
                groups as i32,
                rows as i32,
            )
        })
    }
    pub(crate) fn pointwise(
        &self,
        b: Option<&Self>,
        y: &Self,
        n: usize,
        op: i32,
        scale: f32,
    ) -> Result<()> {
        self.same(&[y])?;
        self.fits(n)?;
        y.fits(n)?;
        ensure!((0..=3).contains(&op), "invalid pointwise op");
        if let Some(b) = b {
            self.same(&[b])?;
            b.fits(n)?;
        } else {
            ensure!(op < 2, "missing operand");
        }
        // SAFETY: all buffers and operation requirements checked.
        checked(unsafe {
            px_pointwise(
                self.session(),
                self.ptr(),
                b.map_or(std::ptr::null(), |b| b.ptr().cast_const()),
                y.ptr(),
                n as i32,
                op,
                scale,
            )
        })
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Explicit strides and offsets allow checking both matrix extents"
    )]
    pub(crate) fn rows(
        &self,
        y: &Self,
        rows: usize,
        width: usize,
        xs: usize,
        ys: usize,
        xo: usize,
        yo: usize,
    ) -> Result<()> {
        ensure!(
            xo + width <= xs && yo + width <= ys,
            "row copy exceeds stride"
        );
        self.same(&[y])?;
        self.fits(rows * xs)?;
        y.fits(rows * ys)?;
        // SAFETY: row extents and buffers checked; caller provides distinct regions.
        checked(unsafe {
            px_rows(
                self.session(),
                self.ptr(),
                y.ptr(),
                rows as i32,
                width as i32,
                xs as i32,
                ys as i32,
                xo as i32,
                yo as i32,
            )
        })
    }
    pub(crate) fn phone_norm(
        &self,
        w: &Self,
        b: &Self,
        y: &Self,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<()> {
        self.same(&[w, b, y])?;
        self.fits(rows * width)?;
        y.fits(rows * width)?;
        w.fits(width)?;
        b.fits(width)?;
        // SAFETY: affine and matrix bounds checked.
        checked(unsafe {
            px_norm(
                self.session(),
                self.ptr(),
                w.ptr(),
                b.ptr(),
                y.ptr(),
                rows as i32,
                width as i32,
                eps,
            )
        })
    }
    pub(crate) fn phone_softmax(&self, y: &Self, rows: usize, width: usize) -> Result<()> {
        ensure!(width > 0 && width <= 512, "unsupported CTC width");
        self.same(&[y])?;
        self.fits(rows * width)?;
        y.fits(rows * width)?;
        // SAFETY: kernel supports the validated vocabulary extent.
        checked(unsafe {
            px_softmax(
                self.session(),
                self.ptr(),
                y.ptr(),
                rows as i32,
                width as i32,
            )
        })
    }
}
