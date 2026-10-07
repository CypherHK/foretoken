// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Frontend-owned routing reservations and request-specific load projections.

use crate::{RouteCandidate, RouteTargetId, RouterRequest};
use foretoken_kv_indexer::KvPrefixIndexer;
use foretoken_model_protocol::ModelServerRole;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

/// Shared routing reservations retained by RuntimeBuilder across serving-snapshot updates.
/// Sessions from retiring generations continue releasing their own reservations into this state.
#[derive(Clone, Default)]
pub struct RoutingLoadState(pub(crate) Arc<Mutex<RoutingReservations>>);

pub(crate) type ReservationKey = (RouteTargetId, u32);

/// Snapshot of this frontend's reservations for one route target and data-parallel rank.
/// Engine telemetry is deliberately not added to these values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoutingLoadSnapshot {
    /// Requests reserved by a selected stage or response stream.
    pub requests: i64,
    /// Uncached prompt tokens reserved by this frontend.
    pub tokens: i64,
}

#[derive(Default)]
pub(crate) struct RoutingReservations {
    requests: BTreeMap<ReservationKey, BTreeMap<String, usize>>,
    // Observation-only full prompt ledger. This never feeds routing scores.
    pending_prompts: BTreeMap<ReservationKey, BTreeMap<String, (usize, bool)>>,
}

impl RoutingReservations {
    /// Takes a coherent candidate snapshot under the routing transaction's lock.
    pub(crate) fn snapshot(&self, key: &ReservationKey) -> RoutingLoadSnapshot {
        let Some(requests) = self.requests.get(key) else {
            return RoutingLoadSnapshot::default();
        };
        RoutingLoadSnapshot {
            requests: requests.len() as i64,
            tokens: requests.values().fold(0_i64, |tokens, request| {
                tokens.wrapping_add(*request as i64)
            }),
        }
    }

    /// Reserves a selected stage before dispatch; the owning session or stream must release it.
    pub(crate) fn reserve(
        &mut self,
        request: &RouterRequest,
        candidate: &RouteCandidate,
        kv: &dyn KvPrefixIndexer,
    ) {
        let key = (
            candidate.route_target_id.clone(),
            candidate.data_parallel_rank,
        );
        // Reserve uncached prompt tokens for each generation stage until its first response.
        // Encoder has no token-generation workload.
        let uncached_tokens = if candidate.role == ModelServerRole::Encoder {
            0
        } else {
            uncached_tokens(request, candidate, kv)
        };
        self.requests
            .entry(key)
            .or_default()
            .insert(request.request_id().to_owned(), uncached_tokens);
    }

    /// Records an Aggregate selection separately from the legacy uncached-token reservation.
    pub(crate) fn reserve_pending_prompt(
        &mut self,
        request: &RouterRequest,
        candidate: &RouteCandidate,
    ) {
        if candidate.role == ModelServerRole::Aggregate && request.generate_request.is_some() {
            self.pending_prompts
                .entry((
                    candidate.route_target_id.clone(),
                    candidate.data_parallel_rank,
                ))
                .or_default()
                .insert(
                    request.request_id().to_owned(),
                    (request.token_count(), false),
                );
        }
    }

    /// Marks frontend dispatch start, including the pending backend generate future.
    pub(crate) fn prompt_dispatched(&mut self, key: &ReservationKey, id: &str) {
        if let Some((_, dispatched)) = self
            .pending_prompts
            .get_mut(key)
            .and_then(|m| m.get_mut(id))
        {
            *dispatched = true;
        }
    }

    /// Sums full inputs still awaiting a nonempty backend token item for this exact target/rank.
    pub(crate) fn pending_prompt_tokens(&self, key: &ReservationKey) -> usize {
        self.pending_prompts.get(key).map_or(0, |m| {
            m.values()
                .filter(|(_, dispatched)| *dispatched)
                .map(|(tokens, _)| *tokens)
                .sum()
        })
    }

    /// Removes the observation-only contribution on first token, failure, completion or cancellation.
    pub(crate) fn release_pending_prompt(&mut self, key: &ReservationKey, id: &str) {
        if let Some(prompts) = self.pending_prompts.get_mut(key) {
            prompts.remove(id);
            if prompts.is_empty() {
                self.pending_prompts.remove(key);
            }
        }
    }

    /// Releases prompt-token load when the first response reaches the frontend.
    pub(crate) fn release_prompt_load(&mut self, key: &ReservationKey, id: &str) {
        if let Some(request) = self
            .requests
            .get_mut(key)
            .and_then(|requests| requests.get_mut(id))
        {
            *request = 0;
        }
    }

    /// Removes exactly one request-stage contribution on completion, rejection, or cancellation.
    pub(crate) fn release(&mut self, key: &ReservationKey, id: &str) {
        self.release_pending_prompt(key, id);
        if let Some(requests) = self.requests.get_mut(key) {
            requests.remove(id);
            if requests.is_empty() {
                self.requests.remove(key);
            }
        }
    }
}

/// Returns uncached indexed tokens and the partial prompt tail for routing load accounting.
pub(crate) fn uncached_tokens(
    request: &RouterRequest,
    candidate: &RouteCandidate,
    kv: &dyn KvPrefixIndexer,
) -> usize {
    let Some(info) = crate::cache::cache_match(request, candidate, kv) else {
        return request.token_count();
    };
    let indexed = info.total_blocks * info.block_size;
    indexed.saturating_sub(info.matched_blocks * info.block_size)
        + request.token_count().saturating_sub(indexed)
}
