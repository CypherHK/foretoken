// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Local writer lifecycle: private output, no overwrite and visible shutdown loss counters.

use foretoken_tracing::RequestObservations;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// The first-row anchor must calibrate the same clock as later rows without breaking drain or loss accounting.
#[test]
fn local_writer_drains_and_protects_existing_output() {
    let path = std::env::temp_dir().join(format!(
        "foretoken-observation-{}-{}.jsonl",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let wall_before = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let (sink, writer) = RequestObservations::create(&path).unwrap();
    let observed_offset_ns = (sink.offset_ms(Instant::now()).unwrap() * 1_000_000.0) as u128;
    let wall_after = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    for i in 0..16 {
        sink.record(&serde_json::json!({"request_id": i}));
    }
    let counts = writer.finish();
    assert_eq!(
        (counts.written, counts.dropped, counts.write_errors),
        (17, 0, 0)
    );
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 17);
    let anchor = &rows[0];
    assert_eq!(anchor["schema_version"], "stage2-observation-v1");
    assert_eq!(anchor["event_type"], "clock_anchor");
    assert_eq!(anchor["clock_domain"], sink.clock_domain());
    assert_eq!(anchor["origin_offset_ms"], 0.0);
    assert!(anchor.get("request_id").is_none());
    assert_eq!(
        rows.iter()
            .filter(|row| row["event_type"] == "clock_anchor")
            .count(),
        1
    );
    let origin = anchor["origin_unix_ns"].as_u64().unwrap() as u128;
    let uncertainty = anchor["wall_sampling_span_ns"].as_u64().unwrap() as u128;
    let projected = origin + observed_offset_ns;
    // Offsets are consumed in milliseconds; SystemTime precision is platform-dependent.
    let tolerance_ns = uncertainty + 1_000_000;
    assert!(projected + tolerance_ns >= wall_before);
    assert!(projected <= wall_after + tolerance_ns);
    assert_eq!(rows[1]["request_id"], 0);
    assert_eq!(rows[16]["request_id"], 15);
    assert_eq!(
        RequestObservations::create(&path).err().unwrap().kind(),
        std::io::ErrorKind::AlreadyExists
    );
    sink.record(&serde_json::json!({"request_id": 17}));
    assert_eq!(sink.counts().dropped, 1);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    std::fs::remove_file(path).unwrap();
}
