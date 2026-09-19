use super::*;
use safetensors::tensor::TensorView;
use safetensors::tensor::serialize_to_file;

#[test]
fn fp16_conversion_handles_aligned_and_unaligned_tensor_data() {
    let source: Vec<_> = (0..512)
        .map(|i| f16::from_f32(i as f32 * 0.125 - 20.))
        .collect();
    let bytes: Vec<_> = source
        .iter()
        .flat_map(|f| f.to_bits().to_le_bytes())
        .collect();
    let expected: Vec<_> = source.iter().map(|f| f.to_f32()).collect();
    for prefix in [0, 1] {
        let mut padded = vec![0; prefix];
        padded.extend_from_slice(&bytes);
        let mut values = vec![42.];
        extend_f16(&mut values, &padded[prefix..]);
        assert_eq!(values[0], 42.);
        assert_eq!(&values[1..], expected);
    }
}

struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Result<Self> {
        let suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "whisper-native-loader-{}-{suffix}",
            std::process::id()
        ));
        std::fs::create_dir(&root)?;
        Ok(Self(root))
    }
    fn index(&self, map: serde_json::Value) -> Result<()> {
        std::fs::write(
            self.0.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"weight_map": map}))?,
        )?;
        Ok(())
    }
    fn tensor(&self, file: &str, name: &str, dtype: Dtype, bytes: &[u8]) -> Result<()> {
        let n = bytes.len() / (dtype.bitsize() / 8);
        serialize_to_file(
            [(name, TensorView::new(dtype, vec![n], bytes)?)],
            None,
            &self.0.join(file),
        )?;
        Ok(())
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
#[ignore = "requires a CUDA device; exercises real model-file loading"]
fn indexed_weights_preserve_values_and_reject_inconsistent_packages() -> Result<()> {
    let device = Device::new(0, false)?;
    let fixture = Fixture::new()?;
    let a: Vec<_> = [1.5_f32, -2.25]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let b: Vec<_> = [0.5_f32, -4.]
        .into_iter()
        .flat_map(|x| f16::from_f32(x).to_bits().to_le_bytes())
        .collect();
    fixture.tensor("one.safetensors", "a", Dtype::F32, &a)?;
    fixture.tensor("two.safetensors", "b", Dtype::F16, &b)?;
    fixture.index(serde_json::json!({"a":"one.safetensors","b":"two.safetensors"}))?;
    let mut weights = Weights::open(&fixture.0, &device)?;
    let first = weights.take("a", &[2])?;
    let second = weights.take("b", &[2])?;
    drop(weights);
    assert_eq!(first.read(2)?, [1.5, -2.25]);
    assert_eq!(second.read(2)?, [0.5, -4.]);
    drop(first);
    assert_eq!(second.read(2)?, [0.5, -4.]);
    let mut weights = Weights::open(&fixture.0, &device)?;
    assert!(weights.take("a", &[1, 2]).is_err());
    drop(weights);

    for map in [
        serde_json::json!({"a":"two.safetensors","b":"one.safetensors"}),
        serde_json::json!({"a":"one.safetensors","b":"two.safetensors","missing":"one.safetensors"}),
        serde_json::json!({"a":"one.safetensors","wrong":"two.safetensors"}),
        serde_json::json!({"a":"../one.safetensors"}),
        serde_json::json!({}),
    ] {
        fixture.index(map)?;
        assert!(Weights::open(&fixture.0, &device).is_err());
    }
    fixture.index(serde_json::json!({"a":"one.safetensors","b":"two.safetensors"}))?;
    fixture.tensor("two.safetensors", "b", Dtype::F32, &f32::NAN.to_le_bytes())?;
    assert!(Weights::open(&fixture.0, &device).is_err());
    fixture.tensor(
        "two.safetensors",
        "b",
        Dtype::BF16,
        &bf16::from_f32(-8.5).to_bits().to_le_bytes(),
    )?;
    assert_eq!(
        Weights::open(&fixture.0, &device)?
            .take("b", &[1])?
            .read(1)?,
        [-8.5]
    );
    fixture.tensor("model.safetensors", "a", Dtype::F32, &a)?;
    assert!(Weights::open(&fixture.0, &device).is_err());
    Ok(())
}

#[test]
fn invalid_model_dimensions_fail_before_allocation() -> Result<()> {
    let dims: Dims = serde_json::from_value(serde_json::json!({
        "audio":{"n_mels":128,"n_audio_ctx":1500,"n_audio_state":1280,"n_audio_head":20,"n_audio_layer":32},
        "text":{"n_vocab":51866,"n_text_ctx":448,"n_text_state":1280,"n_text_head":20,"n_text_layer":32}
    }))?;
    dims.validate()?;
    let mut invalid = dims.clone();
    invalid.audio.n_audio_head = 0;
    assert!(invalid.validate().is_err());
    let mut invalid = dims.clone();
    invalid.text.n_text_ctx = usize::MAX;
    assert!(invalid.validate().is_err());
    let mut invalid = dims.clone();
    invalid.audio.n_audio_layer = 0;
    assert!(invalid.validate().is_err());
    let mut invalid = dims;
    invalid.text.n_text_state = 512;
    assert!(invalid.validate().is_err());
    Ok(())
}
