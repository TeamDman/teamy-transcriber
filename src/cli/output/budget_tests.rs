use super::*;

#[derive(Facet)]
struct Report {
    value: String,
}

#[test]
fn execution_budget_never_changes_serialized_results() {
    for format in [OutputFormat::Text, OutputFormat::Json, OutputFormat::Csv] {
        let mut expected = None;
        for threshold in [
            None,
            Some(Duration::from_secs(1)),
            Some(Duration::from_secs(100)),
        ] {
            let mut bytes = Vec::new();
            CliOutput::facet(
                Report {
                    value: "fixture".into(),
                },
                threshold,
            )
            .emit_to(Some(format), false, &mut bytes)
            .unwrap();
            if let Some(expected) = &expected {
                assert_eq!(&bytes, expected);
            } else {
                expected = Some(bytes.clone());
            }
            assert!(
                !String::from_utf8(bytes)
                    .unwrap()
                    .contains("elapsed_duration")
            );
        }
    }
}
