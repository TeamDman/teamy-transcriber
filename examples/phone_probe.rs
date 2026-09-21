use eyre::Result;
use eyre::ensure;
use std::path::Path;
use std::time::Instant;
use teamy_whisper_native::phones::PhoneModel;
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(args.len() == 4, "phone_probe MODEL WAV OUTPUT");
    let mut wav = hound::WavReader::open(&args[2])?;
    let spec = wav.spec();
    ensure!(
        spec.channels == 1 && spec.sample_rate == 16000,
        "mono16k required"
    );
    let samples: Vec<f32> = if spec.sample_format == hound::SampleFormat::Float {
        wav.samples::<f32>().collect::<Result<_, _>>()?
    } else {
        wav.samples::<i16>()
            .map(|s| s.map(|v| f32::from(v) / 32768.))
            .collect::<Result<_, _>>()?
    };
    let t = Instant::now();
    let model = PhoneModel::load(Path::new(&args[1]), 0).map_err(|e| eyre::eyre!("{e:#}"))?;
    eprintln!("load {:?}", t.elapsed());
    let t = Instant::now();
    let (logits, frames) = model.logits(&samples).map_err(|e| eyre::eyre!("{e:#}"))?;
    eprintln!("inference {:?}, frames {frames}", t.elapsed());
    std::fs::write(
        &args[3],
        logits
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let t = Instant::now();
    let result = model
        .recognize(&samples)
        .map_err(|e| eyre::eyre!("{e:#}"))?;
    eprintln!("warm {:?}", t.elapsed());
    println!("{}", result.ipa);
    Ok(())
}
