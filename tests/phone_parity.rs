//! Opt-in numerical regression against corrected, independently generated reference logits.
use std::path::PathBuf;
use teamy_whisper_native::phones::PhoneModel;
#[test]
#[ignore = "requires CUDA, PHONE_TEST_MODEL and PHONE_TEST_REFERENCE from tools/phone-reference.py"]
fn native_phone_logits_match_corrected_reference() {
    let root = PathBuf::from(std::env::var("PHONE_TEST_REFERENCE").unwrap());
    let mut wav = hound::WavReader::open(root.join("input.wav")).unwrap();
    let audio: Vec<f32> = wav.samples::<f32>().collect::<Result<_, _>>().unwrap();
    let expected: Vec<f32> = std::fs::read(root.join("reference-logits.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let model = PhoneModel::load(
        &PathBuf::from(std::env::var("PHONE_TEST_MODEL").unwrap()),
        0,
    )
    .unwrap();
    let (actual, frames) = model.logits(&audio).unwrap();
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), frames * 428);
    let maximum = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(maximum < 0.002, "maximum logit error {maximum}");
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0
    };
    let matches = actual
        .chunks_exact(428)
        .zip(expected.chunks_exact(428))
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();
    assert_eq!(matches, frames, "frame-level phone predictions differ");
}
