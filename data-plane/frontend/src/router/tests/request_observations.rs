// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! CPU-only personal-branch observation fixture. Exercises the real PipelineRouter and writer;
//! static route/telemetry inputs supply no engine, inference or model-quality evidence.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use foretoken_model_protocol::{DataParallelTelemetry, ModelServerRole};
use foretoken_router::algorithm::{AllowAllFilter, MaxPicker, TokenLoadScorer, UniformScorer};
use foretoken_router::{
    ModelRouteTable, PipelineRouter, RouteInventory, RouteTarget, RouteTargetId, RouteTargetSet,
    RouteTargetStats, RouteTargetStatsReader, Router, RouterPipeline, RouterRequest, ScalingTarget,
    ScalingTargetKind,
};
use foretoken_tracing::RequestObservations;
use serde_json::Value;

struct Inventory(ModelRouteTable);
impl RouteInventory for Inventory {
    fn model_routes(&self) -> &ModelRouteTable {
        &self.0
    }
    fn is_route_target_healthy(&self, _: &RouteTargetId) -> bool {
        true
    }
}
struct CachedStats;
impl RouteTargetStatsReader for CachedStats {
    fn stats(&self, _: &RouteTargetId, _: Duration) -> Option<RouteTargetStats> {
        Some(RouteTargetStats {
            data_parallel_ranks: vec![DataParallelTelemetry {
                data_parallel_rank: 0,
                scheduler_running_requests: Some(3),
                scheduler_waiting_requests: Some(2),
                kv_cache_usage: Some(0.125),
            }],
            collected_at_unix_ms: 100,
            observed_window: Duration::ZERO,
            running_requests: 99,
            max_concurrent_requests: None,
            scheduler_running_requests: Some(99),
            scheduler_waiting_requests: Some(99),
            kv_cache_usage: Some(0.99),
            prompt_tokens_per_second: None,
            generation_tokens_per_second: None,
            ttft: None,
            tpot: None,
            e2e_latency: None,
        })
    }
}
fn inventory(count: usize) -> Arc<Inventory> {
    Arc::new(Inventory(ModelRouteTable::new(
        (0..count)
            .map(|i| {
                let target = ScalingTarget {
                    service_uid: "fixture".into(),
                    name: format!("target-{i}"),
                    uid: format!("target-{i}"),
                    kind: ScalingTargetKind::Pool,
                };
                RouteTarget {
                    route_target_id: RouteTargetId::new(format!("target-{i}")),
                    admission_targets: RouteTargetSet::new(vec![target.clone()]),
                    target,
                    model: "fixture".into(),
                    revision: "fixture".into(),
                    capabilities: BTreeSet::new(),
                    max_input_tokens: None,
                    ready: true,
                    role: ModelServerRole::Aggregate,
                    pipeline_scope_id: None,
                    data_parallel_size: 1,
                }
            })
            .collect(),
    )))
}
fn request(id: &str, len: usize) -> RouterRequest {
    RouterRequest::new(
        "fixture",
        Arc::new(vllm_llm::GenerateRequest {
            request_id: id.into(),
            prompt_token_ids: vec![7; len],
            sampling_params: Default::default(),
            mm_features: None,
            arrival_time: None,
            cache_salt: Some("DO_NOT_EXPORT_SALT".into()),
            trace_headers: None,
            priority: 0,
            data_parallel_rank: None,
            session_id: None,
            reasoning_parser_kwargs: None,
            lora_request: None,
        }),
    )
}
fn rows(path: &std::path::Path) -> Vec<Value> {
    let text = std::fs::read_to_string(path).unwrap();
    assert!(!text.contains("DO_NOT_EXPORT_SALT"));
    assert!(!text.contains("prompt_token_ids"));
    text.lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

// Protect full pending input against premature release by empty items and leaked cancelled stages.
#[tokio::test]
async fn selection_snapshot_tracks_dispatch_first_token_and_cleanup() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("observations.jsonl");
    let (sink, writer) = RequestObservations::create(&path).unwrap();
    let router = PipelineRouter::with_pipeline(
        inventory(1),
        RouterPipeline::new(
            Arc::new(AllowAllFilter),
            Arc::new(UniformScorer),
            Arc::new(MaxPicker),
        ),
    )
    .with_request_observations(Some(sink))
    .with_route_target_stats_reader(Arc::new(CachedStats));
    let mut first = router.start(request("first", 7)).await;
    first.select_initial().unwrap();
    let mut not_dispatched = router.start(request("not-dispatched", 11)).await;
    not_dispatched.select_initial().unwrap();
    drop(not_dispatched);
    first.dispatch_started();
    let mut pending = router.start(request("pending", 11)).await;
    pending.select_initial().unwrap();
    drop(pending);
    // An empty backend item releases legacy load, but must retain the full W_A contribution.
    first.response_started();
    let mut metadata = router.start(request("metadata", 11)).await;
    metadata.select_initial().unwrap();
    drop(metadata);
    first.first_token_received();
    let mut streaming = router.start(request("streaming", 11)).await;
    streaming.select_initial().unwrap();
    drop(streaming);
    drop(first);
    let mut cleaned = router.start(request("cleaned", 11)).await;
    cleaned.select_initial().unwrap();
    drop(cleaned);
    let counts = writer.finish();
    assert_eq!(
        (counts.written, counts.dropped, counts.write_errors),
        (6, 0, 0)
    );
    let rows = rows(&path);
    let expected = [
        (0, 0, 0),
        (1, 7, 0),
        (1, 7, 7),
        (1, 0, 7),
        (1, 0, 0),
        (0, 0, 0),
    ];
    for (row, (inflight, reserved, pending)) in rows.iter().zip(expected) {
        assert_eq!(row["inflight_before"], inflight);
        assert_eq!(row["reserved_tokens"], reserved);
        assert_eq!(row["pending_first_token_input_tokens"], pending);
        assert_eq!(row["features_at_offset_ms"], row["on_selected_offset_ms"]);
        assert_eq!(row["router_rank_scheduler_running_requests"], 3);
        assert_eq!(row["router_rank_scheduler_waiting_requests"], 2);
        assert_eq!(row["router_rank_kv_cache_usage"], 0.125);
        assert!(row.get("received_ms").is_none());
        for field in [
            "prefix_hit_fraction",
            "prefix_matched_blocks",
            "prefix_total_blocks",
        ] {
            assert!(row[field].is_null());
            assert_eq!(row["missing_fields"][field], "kv_index_disabled");
        }
    }
}

// Replay identical live reservations in both modes so the hook cannot influence picker decisions.
#[tokio::test]
async fn enabling_observations_preserves_routing_and_reports_cpu_cost() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("observations.jsonl");
    let (sink, writer) = RequestObservations::create(&path).unwrap();
    let pipeline = Arc::new(RouterPipeline::new(
        Arc::new(AllowAllFilter),
        Arc::new(TokenLoadScorer::default()),
        Arc::new(MaxPicker),
    ));
    let off = PipelineRouter::with_pipeline(inventory(2), pipeline.clone());
    let on =
        PipelineRouter::with_pipeline(inventory(2), pipeline).with_request_observations(Some(sink));
    let mut durations = [Vec::new(), Vec::new()];
    // Keep concurrent live sessions and exercise metadata/first-token/release transitions identically.
    for block in 0..20 {
        let mut sessions = [Vec::new(), Vec::new()];
        let mut decisions = [Vec::new(), Vec::new()];
        for i in 0..50 {
            let id = format!("sample-{block}-{i}");
            for index in [block % 2, 1 - block % 2] {
                let router = if index == 0 { &off } else { &on };
                let mut session = router.start(request(&id, 1 + (i * 31) % 128)).await;
                let started = Instant::now();
                let selected = session.select_initial().unwrap();
                durations[index].push(started.elapsed().as_secs_f64() * 1000.0);
                session.dispatch_started();
                if i % 3 == 0 {
                    session.response_started();
                }
                if i % 5 == 0 {
                    session.first_token_received();
                }
                decisions[index].push((selected.route_target_id, selected.data_parallel_rank));
                sessions[index].push(session);
            }
        }
        assert_eq!(decisions[0], decisions[1]);
        // Sessions drop between balanced blocks, releasing both ledgers.
    }
    let counts = writer.finish();
    assert_eq!(
        (counts.written, counts.dropped, counts.write_errors),
        (1000, 0, 0)
    );
    let rows = rows(&path);
    let costs: Vec<f64> = rows
        .iter()
        .map(|row| row["feature_maintenance_cost_ms"].as_f64().unwrap())
        .collect();
    let output = serde_json::json!({"fixture": "cpu-router-no-engine", "samples_per_mode": 1000,
        "routing_decisions_equal": true, "writer": counts,
        "lock_extra_off_ms": {"p50": 0.0, "p95": 0.0, "basis": "disabled observation block does not execute"},
        "lock_extra_on_ms": {"p50": percentile(&costs, 0.5), "p95": percentile(&costs, 0.95)},
        "selection_off_ms": {"p50": percentile(&durations[0], 0.5), "p95": percentile(&durations[0], 0.95)},
        "selection_on_ms": {"p50": percentile(&durations[1], 0.5), "p95": percentile(&durations[1], 0.95)},
        "measurement_scope": "snapshot construction plus observation-only reserve within routing lock; excludes legacy reserve, lock wait and out-of-lock serialization"});
    if let Some(path) = std::env::var_os("FORETOKEN_T1_FIXTURE_OUTPUT") {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        use std::io::Write;
        writeln!(file, "{}", serde_json::to_string_pretty(&output).unwrap()).unwrap();
    }
    println!("T1_FIXTURE {}", output);
}
fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let position = (sorted.len() - 1) as f64 * q;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (position - lo as f64)
}
