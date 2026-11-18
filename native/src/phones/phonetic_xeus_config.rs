use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use facet::Facet;
use facet_json::RawJson;
use std::path::Path;

#[derive(Facet)]
struct VersionHeader {
    // Upstream checkpoints omit the version field.
    #[facet(default = RawJson::new("0"))]
    schema_version: RawJson<'static>,
}

#[derive(Facet)]
struct ConfigV0 {
    model_type: String,
    sampling_rate: usize,
    interctc_layer_idx: Vec<usize>,
    interctc_use_conditioning: bool,
}

pub(super) struct PhoneticXeusConfigCodec;

impl PhoneticXeusConfigCodec {
    pub(super) fn validate_file(path: &Path) -> Result<()> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading PhoneticXeus config {}", path.display()))?;
        Self::validate(&bytes)
            .with_context(|| format!("validating PhoneticXeus config {}", path.display()))
    }

    fn validate(bytes: &[u8]) -> Result<()> {
        let header: VersionHeader = facet_json::from_slice(bytes)?;
        let version = header
            .schema_version
            .as_str()
            .trim()
            .parse::<u32>()
            .context("PhoneticXeus config schema_version must be an unsigned JSON integer")?;
        match version {
            0 => {
                let config: ConfigV0 = facet_json::from_slice(bytes)?;
                config.validate()
            }
            _ => bail!("unsupported PhoneticXeus config schema_version {version}"),
        }
    }
}

impl ConfigV0 {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.model_type == "phoneticxeus"
                && self.sampling_rate == 16000
                && self.interctc_layer_idx == [4, 8, 12]
                && self.interctc_use_conditioning,
            "unsupported PhoneticXeus architecture/conditioning configuration"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_unversioned_upstream_config_with_extra_metadata() -> Result<()> {
        PhoneticXeusConfigCodec::validate(
            br#"{
                "model_type": "phoneticxeus",
                "sampling_rate": 16000,
                "interctc_layer_idx": [4, 8, 12],
                "interctc_use_conditioning": true,
                "architectures": ["PhoneticXeus"]
            }"#,
        )
    }

    #[test]
    fn rejects_unknown_and_malformed_versions_before_decoding_payload() {
        for (bytes, expected) in [
            (
                br#"{"schema_version":2}"#.as_slice(),
                "unsupported PhoneticXeus config schema_version 2",
            ),
            (
                br#"{"schema_version":null}"#.as_slice(),
                "schema_version must be an unsigned JSON integer",
            ),
        ] {
            let error = PhoneticXeusConfigCodec::validate(bytes).unwrap_err();
            assert!(error.to_string().contains(expected));
        }
    }
}
