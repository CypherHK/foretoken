// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Complete-block prefix observations for request-specific routing costs.

use crate::{RouteCandidate, RouterRequest};
use foretoken_kv_indexer::{
    KvPrefixIndexer, KvPrefixMatch, KvPrefixQueryResult, KvPrefixUnavailableReason,
};
use foretoken_model_protocol::{KvCacheLocality, ModelServerRole};

pub(crate) struct CacheMatch {
    pub(crate) matched_blocks: usize,
    pub(crate) total_blocks: usize,
    pub(crate) block_size: usize,
}

/// Reads complete-block counts from the exact route binding, preserving block units.
pub(crate) fn cache_match(
    request: &RouterRequest,
    candidate: &RouteCandidate,
    kv: &dyn KvPrefixIndexer,
) -> Option<CacheMatch> {
    cache_observation(request, candidate, kv).ok()
}

/// Reads the same complete-block projection, retaining unavailable reasons for local observations.
pub(crate) fn cache_observation(
    request: &RouterRequest,
    candidate: &RouteCandidate,
    kv: &dyn KvPrefixIndexer,
) -> Result<CacheMatch, &'static str> {
    if candidate.role == ModelServerRole::Encoder {
        return Err("encoder_has_no_prefix_semantics");
    }
    let lookup = request
        .kv_prefix_lookup(
            candidate.route_target_id.as_str(),
            candidate.data_parallel_rank,
        )
        .map_err(unavailable_reason)?;
    let matches = match kv.prefix_matches(lookup) {
        KvPrefixQueryResult::Matches(matches) => matches,
        KvPrefixQueryResult::Unavailable(reason) => return Err(unavailable_reason(reason)),
    };
    complete_blocks(request.token_count(), matches.block_size(), matches)
}

/// Projects observed prefix facts into complete blocks; shared by routing costs and observations.
pub(crate) fn complete_blocks(
    input_tokens: usize,
    block_size: Option<u32>,
    matches: impl IntoIterator<Item = KvPrefixMatch>,
) -> Result<CacheMatch, &'static str> {
    let block_size = block_size
        .filter(|size| *size > 0)
        .ok_or("block_size_unavailable")? as usize;
    let mut info = CacheMatch {
        matched_blocks: 0,
        total_blocks: input_tokens / block_size,
        block_size,
    };
    for matched in matches {
        if matched.placement.locality == KvCacheLocality::Unspecified {
            continue;
        }
        let count = (matched.matched_tokens / block_size).min(info.total_blocks);
        info.matched_blocks = info.matched_blocks.max(count);
    }
    Ok(info)
}

fn unavailable_reason(reason: KvPrefixUnavailableReason) -> &'static str {
    match reason {
        KvPrefixUnavailableReason::Disabled => "kv_index_disabled",
        KvPrefixUnavailableReason::SourceUnhealthy => "kv_source_unhealthy",
        KvPrefixUnavailableReason::MissingBinding => "kv_binding_missing",
        KvPrefixUnavailableReason::UnsupportedRequest => "unsupported_request",
        KvPrefixUnavailableReason::RankMismatch => "kv_rank_mismatch",
    }
}
