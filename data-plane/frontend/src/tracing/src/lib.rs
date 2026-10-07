// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Foretoken's compatibility boundary for vLLM tracing initialization.

pub use vllm_tracing::*;

pub mod request_observations;
pub use request_observations::{ObservationCounts, ObservationWriter, RequestObservations};
