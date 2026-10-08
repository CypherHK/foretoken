// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! CPU loopback fixture: real HTTP/decoding/runtime/router/writer with a task-authorized stub LlmFacade.
//! This exercises observation contracts, never inference quality or deployed serving behavior.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use foretoken_chat::{
    ChatRequestProcessor, HfChatBackend, LoadModelBackendsOptions, ParserSelection,
};
use foretoken_llm_facade::{LlmFacade, LlmFacadeError, LlmFacadeResolver, RouteStage, TokenStream};
use foretoken_model_protocol::ModelServerRole;
use foretoken_router::algorithm::{AllowAllFilter, MaxPicker, UniformScorer};
use foretoken_router::{
    ModelRouteTable, PipelineRouter, RouteDecision, RouteInventory, RouteTarget, RouteTargetId,
    RouteTargetSet, RouterPipeline, ScalingTarget, ScalingTargetKind,
};
use foretoken_server::{
    Generation, GenerationRequest, ModelRuntime, RuntimeBundle, RuntimeControl, RuntimeGeneration,
    RuntimeState,
};
use foretoken_text::backend::hf::{HfTextBackend, ResolvedModelFiles, TokenizerSource};
use foretoken_text::{Prompt, TextBackend, TextRequestProcessor};
use foretoken_tracing::RequestObservations;
use futures::StreamExt;
use serde_json::{Value, json};
use vllm_llm::{FinishReason, GenerateOutput, GeneratePromptInfo, GenerateRequest};

struct Inventory(ModelRouteTable);
impl RouteInventory for Inventory {
    fn model_routes(&self) -> &ModelRouteTable {
        &self.0
    }
    fn is_route_target_healthy(&self, _: &RouteTargetId) -> bool {
        true
    }
}
struct StubBackend(Arc<AtomicBool>);
fn output(
    request: &GenerateRequest,
    first: bool,
    ids: &[u32],
    finish: Option<FinishReason>,
) -> GenerateOutput {
    GenerateOutput {
        request_id: request.request_id.clone(),
        prompt_info: first.then(|| GeneratePromptInfo {
            prompt_token_ids: request.prompt_token_ids.clone().into(),
            prompt_logprobs: None,
        }),
        token_ids: ids.to_vec(),
        logprobs: None,
        finish_reason: finish,
        cached_token_count: 2,
        kv_transfer_params: None,
        ec_transfer_params: None,
    }
}
#[async_trait]
impl LlmFacade for StubBackend {
    async fn generate(&self, request: GenerateRequest) -> Result<TokenStream, LlmFacadeError> {
        let mode = request.prompt_token_ids[0];
        if mode == 7 {
            return Err(LlmFacadeError::RequestFailed);
        }
        if mode == 8 {
            self.0.store(true, Ordering::SeqCst);
            return std::future::pending().await;
        }
        Ok(Box::pin(async_stream::stream! {
            yield Ok(output(&request, true, &[], None));
            tokio::time::sleep(Duration::from_millis(8)).await;
            if mode == 11 || mode == 12 {
                // The decoder must count only [a,b] after matching b inside a three-token batch.
                // Cover both an unfinished backend and a match inside its terminal item.
                yield Ok(output(&request, false, &[1, 2, 3],
                    (mode == 12).then_some(FinishReason::Length)));
                if mode == 11 { std::future::pending::<()>().await; }
                return;
            }
            if mode == 10 {
                yield Ok(output(&request, false, &[], Some(FinishReason::Length)));
                return;
            }
            if mode == 2 {
                yield Ok(output(&request, false, &[1], Some(FinishReason::Length)));
                return;
            }
            yield Ok(output(&request, false, &[1, 2], None));
            tokio::time::sleep(Duration::from_millis(8)).await;
            match mode {
                3 => yield Err(LlmFacadeError::RequestFailed),
                4 => {}, // Deliberate nonterminal EOF must remain incomplete, not a normal label.
                5 | 6 => std::future::pending::<()>().await,
                9 => yield Ok(output(&request, false, &[], Some(FinishReason::Abort))),
                _ => yield Ok(output(&request, false, &[3], Some(FinishReason::Length))),
            }
        }))
    }
    async fn abort(&self, _: &[String]) -> Result<(), LlmFacadeError> {
        Ok(())
    }
}
struct Resolver(Arc<AtomicBool>);
impl LlmFacadeResolver for Resolver {
    fn resolve_stage(&self, _: &RouteDecision, _: RouteStage) -> Option<Arc<dyn LlmFacade>> {
        Some(Arc::new(StubBackend(self.0.clone())))
    }
    fn bootstrap_endpoint(&self, _: &RouteDecision) -> Option<String> {
        None
    }
}
struct Ready;
#[async_trait]
impl RuntimeControl for Ready {
    async fn refresh_backend_readiness(&self) {}
    fn configured_models(&self) -> Vec<String> {
        vec!["fixture".into()]
    }
    fn is_ready(&self) -> bool {
        true
    }
}

fn runtime(
    dir: &std::path::Path,
    sink: Option<RequestObservations>,
    dispatch_pending: Arc<AtomicBool>,
) -> Arc<RuntimeGeneration> {
    // Tiny local model metadata and a real HF tokenizer. No model files, network or fake tokenizer API.
    let tokenizer_path = dir.join("tokenizer.json");
    std::fs::write(
        &tokenizer_path,
        json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},
        "post_processor":null,"decoder":null,"model":{"type":"WordLevel","unk_token":"[UNK]",
        "vocab":{"[UNK]":0,"a":1,"b":2,"c":3,"d":4,"e":5,"f":6,"g":7,"h":8,"i":9,"j":10,"k":11,"l":12}}})
        .to_string(),
    )
    .unwrap();
    let config = dir.join("config.json");
    std::fs::write(&config, r#"{"model_type":"gpt2","vocab_size":13}"#).unwrap();
    let tokenizer_config = dir.join("tokenizer_config.json");
    std::fs::write(
        &tokenizer_config,
        r#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}"}"#,
    )
    .unwrap();
    let files = ResolvedModelFiles {
        tokenizer: TokenizerSource::HuggingFace(tokenizer_path),
        tokenizer_config_path: Some(tokenizer_config),
        generation_config_path: None,
        preprocessor_config_path: None,
        video_preprocessor_config_path: None,
        processor_config_path: None,
        chat_template_path: None,
        config_path: Some(config),
    };
    let text = HfTextBackend::from_resolved_model_files(files.clone(), "fixture".into()).unwrap();
    let tokenizer = text.tokenizer();
    let chat = HfChatBackend::from_resolved_model_files(
        files,
        "fixture".into(),
        LoadModelBackendsOptions {
            language_model_only: true,
            ..Default::default()
        },
        tokenizer.clone(),
    )
    .unwrap();
    let bundle = Arc::new(RuntimeBundle::new(
        Arc::new(TextRequestProcessor::new(Arc::new(text), 1024)),
        tokenizer,
        Arc::new(ChatRequestProcessor::render_only(Arc::new(chat))),
    ));
    let target = ScalingTarget {
        service_uid: "fixture".into(),
        name: "fixture".into(),
        uid: "fixture".into(),
        kind: ScalingTargetKind::Pool,
    };
    let inventory = Arc::new(Inventory(ModelRouteTable::new(vec![RouteTarget {
        route_target_id: RouteTargetId::new("fixture"),
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
    }])));
    let router = PipelineRouter::with_pipeline(
        inventory,
        RouterPipeline::new(
            Arc::new(AllowAllFilter),
            Arc::new(UniformScorer),
            Arc::new(MaxPicker),
        ),
    )
    .with_request_observations(sink.clone());
    let generation = Arc::new(
        RuntimeGeneration::new(Duration::from_millis(250)).with_request_observations(sink),
    );
    assert!(generation.replace_state(
        1,
        Arc::new(RuntimeState::new(
            BTreeMap::from([("fixture".into(), ModelRuntime::new(bundle))]),
            Arc::new(router),
            Arc::new(Resolver(dispatch_pending))
        )),
        Arc::new(Ready)
    ));
    generation
}
fn request(id: &str, mode: u32) -> GenerationRequest {
    GenerationRequest {
        model: "fixture".into(),
        request_id: id.into(),
        prompt: Prompt::TokenIds(vec![mode; 4]),
        sampling_params: Default::default(),
        decode_options: Default::default(),
        intermediate: true,
        priority: 0,
        cache_salt: Some("DO_NOT_EXPORT_SALT".into()),
        session_id: None,
        arrival_time: None,
        started_at: Instant::now(),
        tool_call_parser: ParserSelection::None,
        reasoning_parser: ParserSelection::None,
    }
}
fn read_rows(path: &std::path::Path) -> Vec<Value> {
    let content = std::fs::read_to_string(path).unwrap();
    assert!(!content.contains("DO_NOT_EXPORT_SALT"));
    assert!(!content.contains("prompt_token_ids"));
    content
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
async fn wait_for(path: &std::path::Path, id: &str, event: &str) {
    tokio::time::timeout(Duration::from_secs(2), async {
        loop {
            if read_rows(path)
                .iter()
                .any(|row| row["request_id"] == id && row["event_type"] == event)
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}
fn percentile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let pos = (sorted.len() - 1) as f64 * q;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

// Pair labels through the actual loopback HTTP/SSE consumer, then retain every early-termination category.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn frontend_labels_pair_with_loopback_client_and_retain_incomplete_requests() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("observations.jsonl");
    let (sink, writer) = RequestObservations::create(&path).unwrap();
    let dispatch_pending = Arc::new(AtomicBool::new(false));
    let generation = runtime(dir.path(), Some(sink.clone()), dispatch_pending.clone());
    let app = foretoken_server::router(
        generation.clone(),
        Arc::new(|| vec!["fixture".into()]),
        Duration::from_secs(2),
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = stopped.await;
            })
            .await
            .unwrap();
    });
    let client = reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let mut client_rows = Vec::new();
    for index in 0..27 {
        let mode = match index {
            24 => 11,
            25 => 12,
            26 => 1,
            _ if index % 2 == 0 => 1,
            _ => 2,
        };
        let mut body = json!({
            "model":"fixture","prompt":[mode,mode,mode,mode],"max_tokens":4,"stream":true,
            "stream_options":{"include_usage":true}});
        if index >= 24 {
            body["stop"] = json!(if index == 26 { "z" } else { "b" });
        }
        let started = Instant::now();
        let response = client
            .post(format!("http://{address}/v1/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        let mut bytes = response.bytes_stream();
        let mut pending = String::new();
        let mut id = None;
        let mut first_content = None;
        let mut usage = None;
        let mut finish_reason = None;
        while let Some(chunk) = bytes.next().await {
            pending.push_str(std::str::from_utf8(&chunk.unwrap()).unwrap());
            while let Some(end) = pending.find("\n\n") {
                let event: String = pending.drain(..end + 2).collect();
                for line in event.lines().filter_map(|line| line.strip_prefix("data: ")) {
                    if line == "[DONE]" {
                        continue;
                    }
                    let payload: Value = serde_json::from_str(line).unwrap();
                    assert!(payload.get("error").is_none());
                    if let Some(reason) = payload["choices"][0]["finish_reason"].as_str() {
                        finish_reason = Some(reason.to_owned());
                    }
                    if let Some(value) = payload["id"].as_str() {
                        if let Some(id) = &id {
                            assert_eq!(id, value);
                        } else {
                            id = Some(value.to_owned());
                        }
                    }
                    if payload["choices"].as_array().is_some_and(|choices| {
                        choices.iter().any(|choice| {
                            choice["text"].as_str().is_some_and(|text| !text.is_empty())
                        })
                    }) {
                        first_content
                            .get_or_insert_with(|| started.elapsed().as_secs_f64() * 1000.0);
                    }
                    if payload["usage"].is_object() {
                        usage = Some(payload["usage"].clone());
                    }
                }
            }
        }
        client_rows.push(json!({"request_id":id.unwrap(),"call_start_offset_ms":sink.offset_ms(started).unwrap(),
            "client_ttft_ms":first_content.unwrap(),"client_e2e_ms":started.elapsed().as_secs_f64()*1000.0,
            "expected_output_tokens":if mode==11 || mode==12 {2} else if mode==1 {3} else {1},
            "stop_string_case":match index {24=>Some("early_multi_token_match"),25=>Some("terminal_item_match"),26=>Some("nonmatching_stop_string"),_=>None},
            "finish_reason":finish_reason.unwrap(),"usage":usage.unwrap()}));
    }
    // These controls use known generation IDs, including cases with no HTTP response identity.
    let mut controls = Vec::new();
    for mode in [3, 4, 5, 6, 7, 8, 9, 10] {
        let id = format!("control-{mode}");
        if mode == 8 {
            let owner = generation.clone();
            let pending_id = id.clone();
            let pending =
                tokio::spawn(async move { owner.generate(request(&pending_id, 8)).await });
            tokio::time::timeout(Duration::from_secs(2), async {
                while !dispatch_pending.load(Ordering::SeqCst) {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            // A second selection must see the full input of the dispatched, header-stalled request.
            let generated = generation
                .generate(request("dispatch-probe", 2))
                .await
                .unwrap();
            let mut probe = generated.routed.stream;
            while probe.next().await.is_some() {}
            assert!(pending.await.unwrap().is_err());
            wait_for(&path, &id, "frontend_terminal").await;
        } else {
            match generation.generate(request(&id, mode)).await {
                Ok(generated) => {
                    let mut stream = generated.routed.stream;
                    if mode == 5 || mode == 6 {
                        stream.next().await.unwrap().unwrap(); // Empty metadata.
                        stream.next().await.unwrap().unwrap(); // First true token.
                        wait_for(&path, &id, "frontend_ttft").await;
                        assert!(!read_rows(&path).iter().any(|row| row["request_id"] == id
                            && row["event_type"] == "frontend_terminal"));
                        if mode == 6 {
                            drop(stream);
                        } else {
                            while stream.next().await.is_some() {}
                        }
                    } else {
                        while stream.next().await.is_some() {}
                    }
                }
                Err(error) => assert!(
                    mode == 7 || mode == 8,
                    "unexpected pre-stream error: {error}"
                ),
            }
        }
        wait_for(&path, &id, "frontend_terminal").await;
        let row = read_rows(&path)
            .into_iter()
            .find(|row| row["request_id"] == id && row["event_type"] == "frontend_terminal")
            .unwrap();
        let (expected_status, expected_reason, has_ttft) = match mode {
            3 => ("error", "backend_stream_error", true),
            4 => ("incomplete", "backend_closed_without_terminal", true),
            5 => ("timeout", "request_deadline", true),
            6 => ("cancelled", "consumer_closed_cause_unknown", true),
            7 => ("error", "request_failed", false),
            8 => ("timeout", "request_deadline", false),
            9 => ("cancelled", "backend_abort", true),
            10 => ("completed", "backend_length", false),
            _ => unreachable!(),
        };
        assert_eq!(row["frontend_status"], expected_status);
        assert_eq!(row["frontend_termination_reason"], expected_reason);
        assert_eq!(row["frontend_ttft_ms"].is_number(), has_ttft);
        assert_eq!(row["frontend_ttft_ms"].is_null(), !has_ttft);
        assert_eq!(row["frontend_first_token_offset_ms"].is_number(), has_ttft);
        if mode != 10 {
            assert!(row["frontend_e2e_ms"].is_null());
            assert!(row["frontend_tpot_ms"].is_null());
        }
        if mode == 10 {
            assert!(row["frontend_e2e_ms"].is_number());
            assert_eq!(row["frontend_output_token_count"], 0);
            assert!(row["frontend_tpot_ms"].is_null());
        }
        controls.push(row);
    }
    let _ = stop.send(());
    server.await.unwrap();
    let counts = writer.finish();
    assert_eq!((counts.dropped, counts.write_errors), (0, 0));
    let rows = read_rows(&path);
    assert_eq!(
        rows.iter()
            .filter(|row| row["event_type"] == "frontend_terminal")
            .count(),
        36
    );
    // Old ambiguous keys must be absent from both serialized fields and missing-field reasons.
    for row in rows
        .iter()
        .filter(|row| row["event_type"] != "router_selected")
    {
        for old in [
            "status",
            "termination_reason",
            "first_backend_item_offset_ms",
            "first_token_offset_ms",
            "output_token_count",
            "prompt_token_count",
            "cached_token_count",
            "usage_available_offset_ms",
        ] {
            assert!(row.get(old).is_none(), "old label key: {old}");
            assert!(
                row["missing_fields"].get(old).is_none(),
                "old missing-field key: {old}"
            );
        }
    }
    for row in rows
        .iter()
        .filter(|row| row["event_type"] == "router_selected")
    {
        if row["request_id"] == "dispatch-probe" {
            assert_eq!(row["inflight_before"], 1);
            assert_eq!(row["reserved_tokens"], 4);
            assert_eq!(row["pending_first_token_input_tokens"], 4);
        } else {
            assert_eq!(row["inflight_before"], 0);
            assert_eq!(row["reserved_tokens"], 0);
            assert_eq!(row["pending_first_token_input_tokens"], 0);
        }
    }
    let mut paired = Vec::new();
    let mut differences = Vec::new();
    let mut e2e_differences = Vec::new();
    for client_row in &client_rows {
        let id = &client_row["request_id"];
        let finals: Vec<_> = rows
            .iter()
            .filter(|row| row["request_id"] == *id && row["event_type"] == "frontend_terminal")
            .collect();
        assert_eq!(finals.len(), 1);
        let final_row = finals[0];
        let selected = rows
            .iter()
            .find(|row| row["request_id"] == *id && row["event_type"] == "router_selected")
            .unwrap();
        let early = rows
            .iter()
            .find(|row| row["request_id"] == *id && row["event_type"] == "frontend_ttft")
            .unwrap();
        assert_eq!(final_row["clock_domain"], selected["clock_domain"]);
        assert_eq!(
            final_row["on_selected_offset_ms"],
            selected["on_selected_offset_ms"]
        );
        assert_eq!(final_row["frontend_status"], "completed");
        let matched_stop = matches!(
            client_row["stop_string_case"].as_str(),
            Some("early_multi_token_match" | "terminal_item_match")
        );
        assert_eq!(
            final_row["frontend_termination_reason"],
            if matched_stop {
                "frontend_stop_string"
            } else {
                "backend_length"
            }
        );
        assert_eq!(
            client_row["finish_reason"],
            if matched_stop { "stop" } else { "length" }
        );
        if matched_stop {
            assert_eq!(final_row["frontend_output_token_count"], 2); // Full backend item has three IDs.
            assert!(final_row["frontend_e2e_ms"].is_number());
        }
        assert_eq!(
            final_row["frontend_output_token_count"],
            client_row["expected_output_tokens"]
        );
        assert_eq!(final_row["frontend_usage_prompt_tokens"], 4);
        assert_eq!(final_row["frontend_usage_cached_tokens"], 2);
        assert_eq!(
            client_row["usage"]["completion_tokens"],
            final_row["frontend_output_token_count"]
        );
        assert_eq!(
            client_row["usage"]["prompt_tokens"],
            final_row["frontend_usage_prompt_tokens"]
        );
        assert_eq!(
            client_row["usage"]["prompt_tokens_details"]["cached_tokens"],
            final_row["frontend_usage_cached_tokens"]
        );
        let times = [
            client_row["call_start_offset_ms"].as_f64().unwrap(),
            final_row["frontend_entry_offset_ms"].as_f64().unwrap(),
            final_row["on_selected_offset_ms"].as_f64().unwrap(),
            final_row["frontend_first_backend_item_offset_ms"]
                .as_f64()
                .unwrap(),
            final_row["frontend_first_token_offset_ms"]
                .as_f64()
                .unwrap(),
            final_row["frontend_end_offset_ms"].as_f64().unwrap(),
        ];
        assert!(times.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(times[4] - times[3] > 0.0);
        let ttft = final_row["frontend_ttft_ms"].as_f64().unwrap();
        let e2e = final_row["frontend_e2e_ms"].as_f64().unwrap();
        assert!(
            final_row["frontend_ttft_available_offset_ms"]
                .as_f64()
                .unwrap()
                >= times[4]
        );
        assert!(
            final_row["frontend_e2e_available_offset_ms"]
                .as_f64()
                .unwrap()
                >= times[5]
        );
        assert!(early["frontend_e2e_ms"].is_null());
        assert!(early["frontend_usage_prompt_tokens"].is_null());
        if client_row["expected_output_tokens"] == 1 {
            assert!(final_row["frontend_tpot_ms"].is_null());
        } else {
            assert!(
                (final_row["frontend_tpot_ms"].as_f64().unwrap()
                    - (e2e - ttft)
                        / (client_row["expected_output_tokens"].as_u64().unwrap() - 1) as f64)
                    .abs()
                    < 1e-9
            );
        }
        let difference = client_row["client_ttft_ms"].as_f64().unwrap() - ttft;
        let e2e_difference = client_row["client_e2e_ms"].as_f64().unwrap() - e2e;
        if client_row["stop_string_case"].is_null() {
            differences.push(difference);
            e2e_differences.push(e2e_difference);
        }
        paired.push(json!({"request_id":id,"client_ttft_ms":client_row["client_ttft_ms"],"frontend_ttft_ms":ttft,
            "client_minus_frontend_ttft_ms":difference,"client_e2e_ms":client_row["client_e2e_ms"],"frontend_e2e_ms":e2e,
            "client_minus_frontend_e2e_ms":e2e_difference,"frontend_output_token_count":final_row["frontend_output_token_count"],
            "frontend_termination_reason":final_row["frontend_termination_reason"],
            "stop_string_case":client_row["stop_string_case"],"client_finish_reason":client_row["finish_reason"]}));
    }
    let summary = json!({"fixture":"cpu-loopback-http-with-stub-backend","normal_http_requests":27,"http_requests_without_stop_strings":24,"stop_string_http_requests":3,"paired":paired.len(),
        "pair_rate":paired.len() as f64/27.0,"ttft_difference_ms":{"definition":"client minus frontend","min":percentile(&differences,0.0),
        "p50":percentile(&differences,0.5),"p95":percentile(&differences,0.95),"max":percentile(&differences,1.0)},
        "e2e_difference_ms":{"min":percentile(&e2e_differences,0.0),"p50":percentile(&e2e_differences,0.5),"p95":percentile(&e2e_differences,0.95),"max":percentile(&e2e_differences,1.0)},
        "early_ttft_visible_before_termination":true,"pending_dispatch_input_tokens_verified":4,"terminal_requests":36,"controls":controls,"pairs":paired,"writer":counts,
        "clock_basis":"client and frontend run in one CPU process, share std::time::Instant and writer origin; no cross-process conversion tested"});
    if let Some(path) = std::env::var_os("FORETOKEN_T2_FIXTURE_OUTPUT") {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        writeln!(file, "{}", serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    }
    println!(
        "T2_FIXTURE paired=27/27, TTFT differences {}",
        summary["ttft_difference_ms"]
    );
}

// An external Python client must exercise real HTTP ids/framing and a separate writer clock.
// Opt-in only: no listener, file or backend exists unless the caller supplies a private directory.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_loopback_observation_fixture() {
    use std::io::Write;
    let Some(directory) = std::env::var_os("FORETOKEN_T7_FIXTURE_DIRECTORY") else {
        return;
    };
    let directory = std::path::PathBuf::from(directory);
    let limit_ms: u64 = std::env::var("FORETOKEN_T7_FIXTURE_LIMIT_MS")
        .unwrap()
        .parse()
        .unwrap();
    assert!(limit_ms > 0);
    let (sink, writer) = RequestObservations::create(directory.join("events.jsonl")).unwrap();
    let generation = runtime(&directory, Some(sink), Arc::new(AtomicBool::new(false)));
    let app = foretoken_server::router(
        generation.clone(),
        Arc::new(|| vec!["fixture".into()]),
        Duration::from_secs(2),
    )
    .unwrap()
    // Test-only telemetry serves synthetic gauges; frontend events themselves are real.
    .route(
        "/v1/internal/telemetry",
        axum::routing::get(|| async {
            axum::Json(json!({"version":2,"collected_at_unix_ms":
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
            "accepting":true,"running_requests":0,"scheduler_running_requests":0,
            "scheduler_waiting_requests":0,"kv_cache_usage":0.125,
            "prompt_tokens_total":0,"generation_tokens_total":0}))
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut ready = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("ready.json"))
        .unwrap();
    writeln!(
        ready,
        "{}",
        json!({"host":"127.0.0.1","port":address.port(),
        "model":"fixture","process_id":std::process::id(),"telemetry":"synthetic fixture gauges"})
    )
    .unwrap();
    drop(ready);
    let stop = directory.join("stop");
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::time::timeout(Duration::from_millis(limit_ms), async {
                while !stop.exists() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await;
        })
        .await
        .unwrap();
    drop(generation);
    let counts = writer.finish();
    assert_eq!(counts.dropped, 0);
    assert_eq!(counts.write_errors, 0);
    let mut receipt = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(directory.join("finished.json"))
        .unwrap();
    writeln!(receipt, "{}", json!({"process_id":std::process::id(),"written":counts.written,
        "dropped":counts.dropped,"write_errors":counts.write_errors,"stopped_by_client":directory.join("stop").exists()})).unwrap();
}
