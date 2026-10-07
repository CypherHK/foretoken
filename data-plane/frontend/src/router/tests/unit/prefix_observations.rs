// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Focused block projection and serialized snapshot test. Prefix source facts are synthetic;
//! the existing KV public interface does not expose its private block-size builder.

use std::sync::Arc;

use foretoken_kv_indexer::KvPrefixMatch;
use foretoken_model_protocol::{KvCacheLocality, KvPlacement, KvStorageTier, ModelServerRole};
use foretoken_tracing::RequestObservations;

use crate::{
    RouteCandidate, RouteTargetId, RouteTargetSet, RouterRequest, ScalingTarget, ScalingTargetKind,
};

// Verify real projection/serialization for readable hits, a zero denominator and invalid block size;
// avoid widening the KV API solely to make a mock attach its otherwise-private block granularity.
#[test]
fn prefix_block_facts_preserve_units_fraction_and_unavailable_reasons() {
    let dir = tempfile::tempdir().unwrap();
    let (sink, writer) =
        RequestObservations::create(dir.path().join("observations.jsonl")).unwrap();
    let target = ScalingTarget {
        service_uid: "fixture".into(),
        name: "fixture".into(),
        uid: "fixture".into(),
        kind: ScalingTargetKind::Pool,
    };
    let candidate = RouteCandidate {
        route_target_id: RouteTargetId::new("fixture"),
        target: target.clone(),
        admission_targets: RouteTargetSet::new(vec![target]),
        role: ModelServerRole::Aggregate,
        model: "fixture".into(),
        revision: "fixture".into(),
        pipeline_scope_id: None,
        data_parallel_rank: 0,
        route_target_stats: None,
        local_load: Default::default(),
    };
    for (length, block_size, total, matched, fraction, reason) in [
        (40, 16, Some(2), Some(1), Some(0.5), None),
        (
            7,
            16,
            Some(0),
            Some(0),
            None,
            Some("no_complete_prompt_blocks"),
        ),
        (40, 0, None, None, None, Some("block_size_unavailable")),
    ] {
        let request = RouterRequest::new(
            "fixture",
            Arc::new(vllm_llm::GenerateRequest {
                request_id: format!("prefix-{length}-{block_size}"),
                prompt_token_ids: vec![7; length],
                sampling_params: Default::default(),
                mm_features: None,
                arrival_time: None,
                cache_salt: None,
                trace_headers: None,
                priority: 0,
                data_parallel_rank: None,
                session_id: None,
                reasoning_parser_kwargs: None,
                lora_request: None,
            }),
        );
        let facts = vec![KvPrefixMatch {
            placement: KvPlacement {
                tier: KvStorageTier::Device,
                locality: KvCacheLocality::Local,
            },
            matched_tokens: 16,
        }];
        let cache = crate::cache::complete_blocks(length, Some(block_size), facts);
        let (row, _) = crate::observations::SelectionObservation::from_snapshot(
            &sink,
            &request,
            &candidate,
            &crate::routing_load::RoutingReservations::default(),
            cache,
        );
        let row = serde_json::to_value(row).unwrap();
        assert_eq!(row["prefix_total_blocks"], serde_json::json!(total));
        assert_eq!(row["prefix_matched_blocks"], serde_json::json!(matched));
        assert_eq!(row["prefix_hit_fraction"], serde_json::json!(fraction));
        if let Some(reason) = reason {
            assert_eq!(row["missing_fields"]["prefix_hit_fraction"], reason);
        } else {
            assert!(row["missing_fields"].get("prefix_hit_fraction").is_none());
        }
        assert!(row.get("received_ms").is_none());
        assert!(row.get("block_size").is_none());
        assert!(row.get("scheduler_running_requests").is_none());
        assert!(row.get("kv_cache_usage").is_none());
        assert!(row["router_rank_scheduler_running_requests"].is_null());
    }
    writer.finish();
}
