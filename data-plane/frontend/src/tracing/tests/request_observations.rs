// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Local writer lifecycle: private output, no overwrite and visible shutdown loss counters.

use foretoken_tracing::RequestObservations;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Accepted rows must drain, existing evidence must survive, and post-shutdown losses must be counted.
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
    let (sink, writer) = RequestObservations::create(&path).unwrap();
    assert!(sink.offset_ms(Instant::now()).unwrap() >= 0.0);
    for i in 0..16 {
        sink.record(&serde_json::json!({"request_id": i}));
    }
    let counts = writer.finish();
    assert_eq!(
        (counts.written, counts.dropped, counts.write_errors),
        (16, 0, 0)
    );
    assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 16);
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
