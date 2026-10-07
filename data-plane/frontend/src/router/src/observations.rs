// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Typed routing facts captured before the current request's reservation.

use std::collections::BTreeMap;
use std::time::Instant;

use foretoken_kv_indexer::KvPrefixIndexer;
use foretoken_model_protocol::ModelServerRole;
use foretoken_tracing::RequestObservations;
use serde::Serialize;

use crate::{RouteCandidate, RouterRequest, routing_load::RoutingReservations};

#[derive(Serialize)]
pub(crate) struct SelectionObservation {
    schema_version: &'static str,
    event_type: &'static str,
    request_id: String,
    clock_domain: String,
    features_at_offset_ms: f64,
    on_selected_offset_ms: f64,
    tokenized_input_length: Option<usize>,
    route_target_id: String,
    data_parallel_rank: u32,
    role: ModelServerRole,
    inflight_before: i64,
    reserved_tokens: i64,
    pending_first_token_input_tokens: Option<usize>,
    matched_blocks: Option<usize>,
    total_blocks: Option<usize>,
    block_size: Option<usize>,
    prefix_hit_fraction: Option<f64>,
    scheduler_running_requests: Option<u64>,
    scheduler_waiting_requests: Option<u64>,
    kv_cache_usage: Option<f64>,
    collected_at_unix_ms: Option<u64>,
    received_ms: Option<f64>,
    pub(crate) feature_maintenance_cost_ms: f64,
    missing_fields: BTreeMap<&'static str, &'static str>,
}

impl SelectionObservation {
    /// Called only inside the existing routing transaction; reads no network or file.
    pub(crate) fn capture(
        sink: &RequestObservations,
        request: &RouterRequest,
        candidate: &RouteCandidate,
        reservations: &RoutingReservations,
        kv: &dyn KvPrefixIndexer,
    ) -> (Self, Instant) {
        let mut missing = BTreeMap::new();
        let length = request
            .generate_request
            .as_ref()
            .map(|_| request.token_count());
        if length.is_none() {
            missing.insert("tokenized_input_length", "video_not_tokenized");
        }
        let pending = if candidate.role == ModelServerRole::Aggregate && length.is_some() {
            Some(reservations.pending_prompt_tokens(&(
                candidate.route_target_id.clone(),
                candidate.data_parallel_rank,
            )))
        } else {
            missing.insert("pending_first_token_input_tokens", "aggregate_text_only");
            None
        };
        let prefix = match crate::cache::cache_observation(request, candidate, kv) {
            Ok(info) => Some(info),
            Err(reason) => {
                for field in [
                    "matched_blocks",
                    "total_blocks",
                    "block_size",
                    "prefix_hit_fraction",
                ] {
                    missing.insert(field, reason);
                }
                None
            }
        };
        let fraction = prefix.as_ref().and_then(|info| {
            if info.total_blocks == 0 {
                missing.insert("prefix_hit_fraction", "no_complete_prompt_blocks");
                None
            } else {
                Some(info.matched_blocks as f64 / info.total_blocks as f64)
            }
        });
        let rank = candidate.data_parallel_stats();
        let running = rank.and_then(|r| r.scheduler_running_requests);
        let waiting = rank.and_then(|r| r.scheduler_waiting_requests);
        let kv_usage = rank.and_then(|r| r.kv_cache_usage);
        for (field, absent) in [
            ("scheduler_running_requests", running.is_none()),
            ("scheduler_waiting_requests", waiting.is_none()),
            ("kv_cache_usage", kv_usage.is_none()),
        ] {
            if absent {
                missing.insert(
                    field,
                    if rank.is_some() {
                        "rank_gauge_unavailable"
                    } else {
                        "rank_snapshot_unavailable"
                    },
                );
            }
        }
        missing.insert(
            "received_ms",
            "monotonic_telemetry_receive_time_not_retained",
        );
        let collected_at = candidate
            .route_target_stats
            .as_ref()
            .map(|s| s.collected_at_unix_ms);
        if collected_at.is_none() {
            missing.insert("collected_at_unix_ms", "telemetry_snapshot_unavailable");
        }
        // Stamp after all reads, so no snapshot input is observed after its declared decision time.
        let selected_at = Instant::now();
        let at = sink
            .offset_ms(selected_at)
            .expect("observation origin precedes selection");
        (
            Self {
                schema_version: "stage2-observation-v1",
                event_type: "router_selected",
                request_id: request.request_id().to_owned(),
                clock_domain: sink.clock_domain().to_owned(),
                features_at_offset_ms: at,
                on_selected_offset_ms: at,
                tokenized_input_length: length,
                route_target_id: candidate.route_target_id.as_str().to_owned(),
                data_parallel_rank: candidate.data_parallel_rank,
                role: candidate.role,
                inflight_before: candidate.local_load.requests,
                reserved_tokens: candidate.local_load.tokens,
                pending_first_token_input_tokens: pending,
                matched_blocks: prefix.as_ref().map(|p| p.matched_blocks),
                total_blocks: prefix.as_ref().map(|p| p.total_blocks),
                block_size: prefix.as_ref().map(|p| p.block_size),
                prefix_hit_fraction: fraction,
                scheduler_running_requests: running,
                scheduler_waiting_requests: waiting,
                kv_cache_usage: kv_usage,
                collected_at_unix_ms: collected_at,
                received_ms: None,
                feature_maintenance_cost_ms: 0.0,
                missing_fields: missing,
            },
            selected_at,
        )
    }
}
