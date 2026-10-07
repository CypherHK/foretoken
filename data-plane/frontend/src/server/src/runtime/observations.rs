// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Request-scoped frontend labels. One guard owns early failure, streaming and drop recording.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use foretoken_llm_facade::LlmFacadeError;
use foretoken_tracing::RequestObservations;
use serde::Serialize;
use vllm_llm::{FinishReason, GenerateOutput};

use super::GenerationError;

/// Raw stream termination categories, without response text or backend error payloads.
#[derive(Clone, Copy)]
pub(super) enum Termination {
    Completed(&'static str),
    Error(&'static str),
    Timeout,
    ConsumerClosed,
    Incomplete,
    Dropped,
    Aborted,
}

impl Termination {
    pub(super) fn status_reason(self) -> (&'static str, &'static str) {
        match self {
            Self::Completed(reason) => ("completed", reason),
            Self::Error(reason) => ("error", reason),
            Self::Timeout => ("timeout", "request_deadline"),
            Self::ConsumerClosed => ("cancelled", "consumer_closed_cause_unknown"),
            Self::Incomplete => ("incomplete", "backend_closed_without_terminal"),
            Self::Dropped => ("cancelled", "generation_future_or_task_dropped"),
            Self::Aborted => ("cancelled", "backend_abort"),
        }
    }

    pub(super) fn from_item(item: &Result<GenerateOutput, LlmFacadeError>) -> Option<Self> {
        match item {
            Err(_) => Some(Self::Error("backend_stream_error")),
            Ok(output) => output.finish_reason.as_ref().map(|reason| match reason {
                FinishReason::Stop(_) => Self::Completed("backend_stop"),
                FinishReason::Length => Self::Completed("backend_length"),
                FinishReason::Repetition(_) => Self::Completed("backend_repetition"),
                FinishReason::Abort => Self::Aborted,
                FinishReason::Error => Self::Error("backend_finish_error"),
            }),
        }
    }
}

#[derive(Serialize)]
struct FrontendObservation {
    schema_version: &'static str,
    event_type: &'static str,
    request_id: String,
    clock_domain: String,
    status: &'static str,
    termination_reason: Option<&'static str>,
    frontend_entry_offset_ms: Option<f64>,
    on_selected_offset_ms: Option<f64>,
    first_backend_item_offset_ms: Option<f64>,
    first_token_offset_ms: Option<f64>,
    frontend_end_offset_ms: Option<f64>,
    frontend_ttft_ms: Option<f64>,
    frontend_e2e_ms: Option<f64>,
    frontend_tpot_ms: Option<f64>,
    frontend_ttft_available_offset_ms: Option<f64>,
    frontend_e2e_available_offset_ms: Option<f64>,
    output_token_count: usize,
    // Post-hoc usage is exposed only in the final record, never as a routing feature.
    prompt_token_count: Option<usize>,
    cached_token_count: Option<usize>,
    usage_available_offset_ms: Option<f64>,
    missing_fields: BTreeMap<&'static str, &'static str>,
}

pub(super) struct RequestObservation {
    sink: RequestObservations,
    selected_at: Arc<OnceLock<Instant>>,
    row: FrontendObservation,
    prompt_tokens: Option<usize>,
    cached_tokens: Option<usize>,
    finished: bool,
}

impl RequestObservation {
    /// Generation keeps this guard through preparation, then transfers it to the deadline task.
    pub(super) fn new(sink: RequestObservations, id: String, entry: Instant) -> Self {
        let entry_ms = sink.offset_ms(entry);
        Self {
            selected_at: Arc::new(OnceLock::new()),
            row: FrontendObservation {
                schema_version: "stage2-observation-v1",
                event_type: "frontend_terminal",
                request_id: id,
                clock_domain: sink.clock_domain().to_owned(),
                status: "streaming",
                termination_reason: None,
                frontend_entry_offset_ms: entry_ms,
                on_selected_offset_ms: None,
                first_backend_item_offset_ms: None,
                first_token_offset_ms: None,
                frontend_end_offset_ms: None,
                frontend_ttft_ms: None,
                frontend_e2e_ms: None,
                frontend_tpot_ms: None,
                frontend_ttft_available_offset_ms: None,
                frontend_e2e_available_offset_ms: None,
                output_token_count: 0,
                prompt_token_count: None,
                cached_token_count: None,
                usage_available_offset_ms: None,
                missing_fields: BTreeMap::new(),
            },
            sink,
            prompt_tokens: None,
            cached_tokens: None,
            finished: false,
        }
    }

    /// Shares the first selection timestamp with Router; the guard remains the sole record owner.
    pub(super) fn selected_timestamp(&self) -> Arc<OnceLock<Instant>> {
        self.selected_at.clone()
    }

    pub(super) fn set_prompt_tokens(&mut self, tokens: usize) {
        self.prompt_tokens = Some(tokens);
    }

    fn refresh_selection(&mut self) {
        self.row.on_selected_offset_ms = self
            .selected_at
            .get()
            .and_then(|at| self.sink.offset_ms(*at));
    }

    /// Observes newly generated token IDs, not transport chunks or decoded text length.
    pub(super) fn item(&mut self, item: &Result<GenerateOutput, LlmFacadeError>) {
        let at = self
            .sink
            .offset_ms(Instant::now())
            .expect("origin precedes output");
        self.row.first_backend_item_offset_ms.get_or_insert(at);
        if let Ok(output) = item {
            if let Some(ids) = output.prompt_token_ids() {
                self.prompt_tokens = Some(ids.len());
            }
            self.cached_tokens = Some(
                self.cached_tokens
                    .unwrap_or(0)
                    .max(output.cached_token_count),
            );
            self.row.output_token_count += output.token_ids.len();
            if !output.token_ids.is_empty() && self.row.first_token_offset_ms.is_none() {
                self.row.first_token_offset_ms = Some(at);
                self.row.frontend_ttft_ms =
                    self.row.frontend_entry_offset_ms.map(|entry| at - entry);
                self.refresh_selection();
                self.row.frontend_ttft_available_offset_ms = self.sink.offset_ms(Instant::now());
                self.row.event_type = "frontend_ttft";
                self.refresh_missing_fields();
                self.sink.record(&self.row);
            }
        }
    }

    /// Records a pre-stream failure with its existing stable category; no payload is logged.
    pub(super) fn preparation_error(&mut self, error: GenerationError) {
        let termination = match error {
            GenerationError::DeadlineExceeded => Termination::Timeout,
            GenerationError::InvalidRequest => Termination::Error("invalid_request"),
            GenerationError::ModelNotFound => Termination::Error("model_not_found"),
            GenerationError::Unavailable => Termination::Error("backend_unavailable"),
            GenerationError::BackendRejected => Termination::Error("backend_rejected"),
            GenerationError::BackendProtocol => Termination::Error("backend_protocol"),
            GenerationError::RequestFailed => Termination::Error("request_failed"),
            GenerationError::Internal => Termination::Error("frontend_internal"),
        };
        self.finish(termination);
    }

    /// Finishes exactly once. Failed/cancelled durations remain censored, not valid E2E targets.
    pub(super) fn finish(&mut self, termination: Termination) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.refresh_selection();
        let at = self
            .sink
            .offset_ms(Instant::now())
            .expect("origin precedes termination");
        let (status, reason) = termination.status_reason();
        self.row.event_type = "frontend_terminal";
        self.row.status = status;
        self.row.termination_reason = Some(reason);
        self.row.frontend_end_offset_ms = Some(at);
        if status == "completed" {
            self.row.frontend_e2e_ms = self.row.frontend_entry_offset_ms.map(|entry| at - entry);
            self.row.frontend_e2e_available_offset_ms = self.sink.offset_ms(Instant::now());
            if self.row.output_token_count > 1 {
                self.row.frontend_tpot_ms = self
                    .row
                    .frontend_e2e_ms
                    .zip(self.row.frontend_ttft_ms)
                    .map(|(e2e, ttft)| (e2e - ttft) / (self.row.output_token_count - 1) as f64);
            }
        }
        if self.cached_tokens.is_some() {
            self.row.prompt_token_count = self.prompt_tokens;
            self.row.cached_token_count = self.cached_tokens;
            self.row.usage_available_offset_ms = self.sink.offset_ms(Instant::now());
        }
        self.refresh_missing_fields();
        self.sink.record(&self.row);
    }
    // Both early and terminal rows explain every unavailable label/usage value.
    fn refresh_missing_fields(&mut self) {
        self.row.missing_fields.clear();
        for (field, absent, reason) in [
            (
                "frontend_end_offset_ms",
                self.row.frontend_end_offset_ms.is_none(),
                "stream_not_terminated",
            ),
            (
                "frontend_entry_offset_ms",
                self.row.frontend_entry_offset_ms.is_none(),
                "entry_precedes_clock_origin",
            ),
            (
                "on_selected_offset_ms",
                self.row.on_selected_offset_ms.is_none(),
                "no_observed_router_selection",
            ),
            (
                "first_backend_item_offset_ms",
                self.row.first_backend_item_offset_ms.is_none(),
                "no_backend_item",
            ),
            (
                "first_token_offset_ms",
                self.row.first_token_offset_ms.is_none(),
                "no_nonempty_token_item",
            ),
            (
                "frontend_ttft_ms",
                self.row.frontend_ttft_ms.is_none(),
                "no_first_token_or_entry",
            ),
            (
                "frontend_ttft_available_offset_ms",
                self.row.frontend_ttft_available_offset_ms.is_none(),
                "ttft_unavailable",
            ),
            (
                "frontend_e2e_ms",
                self.row.frontend_e2e_ms.is_none(),
                "incomplete_or_missing_entry",
            ),
            (
                "frontend_e2e_available_offset_ms",
                self.row.frontend_e2e_available_offset_ms.is_none(),
                "e2e_unavailable",
            ),
            (
                "frontend_tpot_ms",
                self.row.frontend_tpot_ms.is_none(),
                "requires_complete_ttft_e2e_and_more_than_one_token",
            ),
            (
                "prompt_token_count",
                self.row.prompt_token_count.is_none(),
                "no_backend_usage",
            ),
            (
                "cached_token_count",
                self.row.cached_token_count.is_none(),
                "no_backend_usage",
            ),
            (
                "usage_available_offset_ms",
                self.row.usage_available_offset_ms.is_none(),
                "no_backend_usage",
            ),
        ] {
            if absent {
                self.row.missing_fields.insert(field, reason);
            }
        }
    }
}

impl Drop for RequestObservation {
    fn drop(&mut self) {
        self.finish(Termination::Dropped);
    }
}
