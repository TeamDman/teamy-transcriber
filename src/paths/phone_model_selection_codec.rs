use super::AppHome;
use super::PhoneModelDir;
use eyre::Context;
use eyre::Result;
use eyre::bail;
use facet::Facet;
use facet_json::RawJson;
use std::io::ErrorKind;
use std::path::PathBuf;

const SELECTION_FILE: &str = "phone-model-selection.json";
const CURRENT_SCHEMA_VERSION: u32 = 1;

#[derive(Facet)]
struct VersionHeader {
    #[facet(default = RawJson::new("0"))]
    schema_version: RawJson<'static>,
}

#[derive(Facet)]
struct SelectionV0 {
    model_dir: String,
}

#[derive(Facet)]
#[facet(deny_unknown_fields)]
struct SelectionV1 {
    schema_version: u32,
    model_dir: String,
}

impl SelectionV0 {
    fn upgrade(self) -> SelectionV1 {
        SelectionV1 {
            schema_version: CURRENT_SCHEMA_VERSION,
            model_dir: self.model_dir,
        }
    }
}

pub(super) struct PhoneModelSelectionCodec;

impl PhoneModelSelectionCodec {
    pub(super) fn read(app_home: &AppHome) -> Result<Option<PhoneModelDir>> {
        let path = app_home.file_path(SELECTION_FILE);
        match std::fs::read(&path) {
            Ok(bytes) => Self::decode(&bytes)
                .map(Some)
                .wrap_err_with(|| format!("invalid phone model selection at {}", path.display())),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).wrap_err_with(|| format!("reading {}", path.display())),
        }
    }

    pub(super) fn write(app_home: &AppHome, model_dir: &PhoneModelDir) -> Result<()> {
        let encoded = Self::encode(model_dir)?;
        app_home.ensure_dir()?;
        let temporary = app_home.file_path(&format!(".phone-model-{}.tmp", uuid::Uuid::new_v4()));
        let result: Result<()> = (|| {
            std::fs::write(&temporary, encoded)?;
            std::fs::rename(&temporary, app_home.file_path(SELECTION_FILE))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result.wrap_err("saving phone model selection")
    }

    fn decode(bytes: &[u8]) -> Result<PhoneModelDir> {
        let header: VersionHeader = facet_json::from_slice(bytes)?;
        let version = header
            .schema_version
            .as_str()
            .trim()
            .parse::<u32>()
            .wrap_err("phone model selection schema_version must be an unsigned JSON integer")?;
        let selection = match version {
            0 => facet_json::from_slice::<SelectionV0>(bytes)?.upgrade(),
            CURRENT_SCHEMA_VERSION => facet_json::from_slice::<SelectionV1>(bytes)?,
            _ => bail!("unsupported phone model selection schema_version {version}"),
        };
        Ok(PhoneModelDir::from(PathBuf::from(selection.model_dir)))
    }

    fn encode(model_dir: &PhoneModelDir) -> Result<String> {
        let model_dir = model_dir
            .to_str()
            .ok_or_else(|| eyre::eyre!("model path is not UTF-8"))?;
        Ok(facet_json::to_string(&SelectionV1 {
            schema_version: CURRENT_SCHEMA_VERSION,
            model_dir: model_dir.to_owned(),
        })?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_legacy_selection_and_writes_current_version() -> Result<()> {
        let model_dir =
            PhoneModelSelectionCodec::decode(br#"{"model_dir":"models\\a model folder"}"#)?;
        let encoded = PhoneModelSelectionCodec::encode(&model_dir)?;
        assert!(encoded.contains(r#""schema_version":1"#));
        assert_eq!(
            PhoneModelSelectionCodec::decode(encoded.as_bytes())?,
            model_dir
        );
        Ok(())
    }

    #[test]
    fn rejects_unknown_or_invalid_schema_versions() {
        let error = PhoneModelSelectionCodec::decode(br#"{"schema_version":2}"#).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported phone model selection schema_version 2")
        );
        for bytes in [
            br#"{"schema_version":null,"model_dir":"model"}"#.as_slice(),
            br#"{"schema_version":"1","model_dir":"model"}"#,
            br#"{"schema_version":1,"model_dir":"model","unknown":true}"#,
        ] {
            let _error =
                PhoneModelSelectionCodec::decode(bytes).expect_err("invalid selection should fail");
        }
    }
}
