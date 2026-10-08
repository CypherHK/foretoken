// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Opt-in release CPU probe of existing observation paths. No listener or engine is started.
//! Counts and local assets are supplied by the caller; timings are descriptive, without gates.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::hint::black_box;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use foretoken_chat::{
    ChatRequestProcessor, HfChatBackend, LoadModelBackendsOptions, ParserSelection,
};
use foretoken_llm_facade::{LlmFacade, LlmFacadeError, LlmFacadeResolver, RouteStage, TokenStream};
use foretoken_model_protocol::{
    CumulativeHistogram, CumulativeHistogramBucket, DataParallelTelemetry, ModelServerRole,
    TelemetryResponse,
};
use foretoken_router::algorithm::{AllowAllFilter, MaxPicker, TokenLoadScorer};
use foretoken_router::{
    ModelRouteTable, PipelineRouter, RouteDecision, RouteInventory, RouteTarget, RouteTargetId,
    RouteTargetSet, RouteTargetStats, RouteTargetStatsReader, Router, RouterPipeline,
    RouterRequest, ScalingTarget, ScalingTargetKind,
};
use foretoken_server::{
    Generation, GenerationRequest, ModelRuntime, RuntimeBundle, RuntimeControl, RuntimeGeneration,
    RuntimeState,
};
use foretoken_text::backend::hf::{HfTextBackend, ResolvedModelFiles, TokenizerSource};
use foretoken_text::{Prompt, TextBackend, TextRequestProcessor};
use foretoken_tokenizer::{HuggingFaceTokenizer, Tokenizer};
use foretoken_tracing::RequestObservations;
use futures::StreamExt;
use serde_json::{Value, json};
use vllm_llm::{FinishReason, GenerateOutput, GeneratePromptInfo, GenerateRequest};

// Reuse the registry's actual history calculation without changing its visibility or polling HTTP.
#[path = "../../backend-registry/src/route_target_stats.rs"]
mod history;

struct Inventory(ModelRouteTable);
impl RouteInventory for Inventory {
    fn model_routes(&self) -> &ModelRouteTable {
        &self.0
    }
    fn is_route_target_healthy(&self, _: &RouteTargetId) -> bool {
        true
    }
}

struct CachedStats(Mutex<history::RouteTargetStatsHistory>, Duration);
impl RouteTargetStatsReader for CachedStats {
    fn stats(&self, _: &RouteTargetId, _: Duration) -> Option<RouteTargetStats> {
        self.0.lock().unwrap().stats(self.1)
    }
}

fn inventory() -> Arc<Inventory> {
    let target = ScalingTarget {
        service_uid: "cpu-probe".into(),
        name: "cpu-probe".into(),
        uid: "cpu-probe".into(),
        kind: ScalingTargetKind::Pool,
    };
    Arc::new(Inventory(ModelRouteTable::new(vec![RouteTarget {
        route_target_id: RouteTargetId::new("cpu-probe"),
        admission_targets: RouteTargetSet::new(vec![target.clone()]),
        target,
        model: "cpu-probe".into(),
        revision: "synthetic".into(),
        capabilities: BTreeSet::new(),
        max_input_tokens: None,
        ready: true,
        role: ModelServerRole::Aggregate,
        pipeline_scope_id: None,
        data_parallel_size: 1,
    }])))
}

fn router(sink: Option<RequestObservations>, stats: Arc<CachedStats>) -> PipelineRouter {
    PipelineRouter::with_pipeline(
        inventory(),
        RouterPipeline::new(
            Arc::new(AllowAllFilter),
            Arc::new(TokenLoadScorer::default()),
            Arc::new(MaxPicker),
        ),
    )
    .with_route_target_stats_reader(stats)
    .with_request_observations(sink)
}

fn request(id: String, tokens: usize) -> GenerateRequest {
    GenerateRequest {
        request_id: id,
        prompt_token_ids: vec![1; tokens],
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
    }
}

/// Holds dispatched sessions across a block, then measures empty-item, true-first and cleanup calls.
async fn router_block(
    router: &PipelineRouter,
    block: &str,
    count: usize,
    tokens: usize,
) -> Vec<Value> {
    let mut sessions = Vec::new();
    let mut samples = Vec::new();
    for index in 0..count {
        let req = RouterRequest::new(
            "cpu-probe",
            Arc::new(request(format!("{block}-{index}"), tokens)),
        );
        let mut session = router.start(req).await;
        let start = Instant::now();
        let selected = session.select_initial().unwrap();
        let selection_ns = start.elapsed().as_nanos();
        assert_eq!(selected.route_target_id, RouteTargetId::new("cpu-probe"));
        assert_eq!(selected.data_parallel_rank, 0);
        let start = Instant::now();
        session.dispatch_started();
        let dispatch_ns = start.elapsed().as_nanos();
        samples.push(json!({"selection_ns":selection_ns,"dispatch_ns":dispatch_ns}));
        sessions.push(session);
    }
    for (session, sample) in sessions.into_iter().zip(&mut samples) {
        let mut session = session;
        let start = Instant::now();
        session.response_started();
        sample["empty_item_callback_ns"] = json!(start.elapsed().as_nanos());
        let start = Instant::now();
        session.first_token_received();
        sample["true_first_callback_ns"] = json!(start.elapsed().as_nanos());
        let start = Instant::now();
        drop(session);
        sample["cleanup_callback_ns"] = json!(start.elapsed().as_nanos());
    }
    // This unmeasured selection exposes any leaked pending load to the emitted observation.
    let mut sentinel = router
        .start(RouterRequest::new(
            "cpu-probe",
            Arc::new(request(format!("{block}-cleanup"), tokens)),
        ))
        .await;
    sentinel.select_initial().unwrap();
    samples
}

struct StubBackend;
#[async_trait]
impl LlmFacade for StubBackend {
    async fn generate(&self, request: GenerateRequest) -> Result<TokenStream, LlmFacadeError> {
        Ok(Box::pin(async_stream::stream! {
            for (index, ids) in [vec![], vec![1, 2], vec![3]].into_iter().enumerate() {
                yield Ok(GenerateOutput {
                    request_id: request.request_id.clone(),
                    prompt_info: (index == 0).then(|| GeneratePromptInfo {
                        prompt_token_ids: request.prompt_token_ids.clone().into(),
                        prompt_logprobs: None,
                    }),
                    token_ids: ids,
                    logprobs: None,
                    finish_reason: (index == 2).then_some(FinishReason::Length),
                    cached_token_count: 0,
                    kv_transfer_params: None,
                    ec_transfer_params: None,
                });
            }
        }))
    }
    async fn abort(&self, _: &[String]) -> Result<(), LlmFacadeError> {
        Ok(())
    }
}
struct Resolver;
impl LlmFacadeResolver for Resolver {
    fn resolve_stage(&self, _: &RouteDecision, _: RouteStage) -> Option<Arc<dyn LlmFacade>> {
        Some(Arc::new(StubBackend))
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
        vec!["cpu-probe".into()]
    }
    fn is_ready(&self) -> bool {
        true
    }
}

/// Builds a socket-free real RuntimeGeneration with tiny metadata and a zero-delay backend stub.
fn runtime(
    dir: &Path,
    sink: Option<RequestObservations>,
    stats: Arc<CachedStats>,
    tokens: usize,
    deadline: Duration,
) -> Arc<RuntimeGeneration> {
    let tokenizer_path = dir.join("tokenizer.json");
    std::fs::write(
        &tokenizer_path,
        json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
            "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,
            "decoder":null,"model":{"type":"WordLevel","unk_token":"[UNK]",
            "vocab":{"[UNK]":0,"a":1,"b":2,"c":3}}})
        .to_string(),
    )
    .unwrap();
    let config = dir.join("config.json");
    std::fs::write(&config, r#"{"model_type":"gpt2","vocab_size":4}"#).unwrap();
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
    let text = HfTextBackend::from_resolved_model_files(files.clone(), "cpu-probe".into()).unwrap();
    let tokenizer = text.tokenizer();
    let chat = HfChatBackend::from_resolved_model_files(
        files,
        "cpu-probe".into(),
        LoadModelBackendsOptions {
            language_model_only: true,
            ..Default::default()
        },
        tokenizer.clone(),
    )
    .unwrap();
    let bundle = Arc::new(RuntimeBundle::new(
        Arc::new(TextRequestProcessor::new(
            Arc::new(text),
            u32::try_from(tokens + 16).unwrap(),
        )),
        tokenizer,
        Arc::new(ChatRequestProcessor::render_only(Arc::new(chat))),
    ));
    let generation =
        Arc::new(RuntimeGeneration::new(deadline).with_request_observations(sink.clone()));
    assert!(generation.replace_state(
        1,
        Arc::new(RuntimeState::new(
            BTreeMap::from([("cpu-probe".into(), ModelRuntime::new(bundle))]),
            Arc::new(router(sink, stats)),
            Arc::new(Resolver),
        )),
        Arc::new(Ready),
    ));
    generation
}

async fn runtime_block(
    runtime: &RuntimeGeneration,
    block: &str,
    count: usize,
    tokens: usize,
) -> Vec<u128> {
    let mut samples = Vec::new();
    for index in 0..count {
        let request = GenerationRequest {
            model: "cpu-probe".into(),
            request_id: format!("{block}-{index}"),
            prompt: Prompt::TokenIds(vec![1; tokens]),
            sampling_params: Default::default(),
            decode_options: Default::default(),
            intermediate: true,
            priority: 0,
            cache_salt: None,
            session_id: None,
            arrival_time: None,
            started_at: Instant::now(),
            tool_call_parser: ParserSelection::None,
            reasoning_parser: ParserSelection::None,
        };
        let start = Instant::now();
        let mut stream = runtime.generate(request).await.unwrap().routed.stream;
        let mut token_ids = Vec::new();
        let mut terminal = false;
        while let Some(item) = stream.next().await {
            let item = item.unwrap();
            token_ids.extend(item.token_ids);
            terminal |= matches!(item.finish_reason, Some(FinishReason::Length));
        }
        samples.push(start.elapsed().as_nanos());
        assert_eq!(token_ids, [1, 2, 3]);
        assert!(terminal);
    }
    samples
}

fn positive_env(name: &str) -> usize {
    let value = std::env::var(name).unwrap().parse::<usize>().unwrap();
    assert!(value > 0, "{name} must be positive");
    value
}

fn finish(writer: foretoken_tracing::ObservationWriter) -> Value {
    let counts = writer.finish();
    assert_eq!(counts.dropped, 0);
    assert_eq!(counts.write_errors, 0);
    json!({"written":counts.written,"dropped":counts.dropped,"write_errors":counts.write_errors})
}

// Direct release measurements cover T1/T2 together and each owner without altering production code.
#[tokio::test(flavor = "current_thread")]
async fn release_cpu_observation_components() {
    let Ok(output) = std::env::var("FORETOKEN_CPU_PROBE_OUTPUT") else {
        return;
    };
    assert!(
        !black_box(cfg!(debug_assertions)),
        "CPU probe requires cargo test --release"
    );
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .unwrap();
    let output = Path::new(&output);
    let blocks = positive_env("FORETOKEN_CPU_PROBE_BLOCKS");
    let per_block = positive_env("FORETOKEN_CPU_PROBE_PER_BLOCK");
    let state_polls = positive_env("FORETOKEN_CPU_PROBE_STATE_POLLS");
    let window = Duration::from_millis(positive_env("FORETOKEN_CPU_PROBE_STATE_WINDOW_MS") as u64);
    let deadline = Duration::from_millis(positive_env("FORETOKEN_CPU_PROBE_DEADLINE_MS") as u64);
    let count = blocks.checked_mul(per_block).unwrap();
    let tokenizer_path = std::env::var("FORETOKEN_CPU_PROBE_TOKENIZER").unwrap();
    let tokenizer = HuggingFaceTokenizer::new_fastokens(Path::new(&tokenizer_path)).unwrap();
    let input =
        std::fs::read_to_string(std::env::var("FORETOKEN_CPU_PROBE_INPUT").unwrap()).unwrap();
    let tokens = tokenizer.encode(&input, false).unwrap();
    let mut tokenizer_ns = Vec::new();
    for _ in 0..count {
        let start = Instant::now();
        black_box(tokenizer.encode(black_box(&input), false).unwrap());
        tokenizer_ns.push(start.elapsed().as_nanos());
    }

    let mut history = history::RouteTargetStatsHistory::new(Duration::MAX);
    for index in 1..=state_polls as u64 {
        let histogram = CumulativeHistogram {
            count: index,
            sum_seconds: index as f64 / 100.0,
            buckets: vec![CumulativeHistogramBucket {
                le_seconds: 1.0,
                count: index,
            }],
        };
        history.push(TelemetryResponse {
            version: 1,
            collected_at_unix_ms: index * 1000,
            accepting: true,
            data_parallel_ranks: vec![DataParallelTelemetry {
                data_parallel_rank: 0,
                scheduler_running_requests: Some(3),
                scheduler_waiting_requests: Some(2),
                kv_cache_usage: Some(0.125),
            }],
            running_requests: 3,
            max_concurrent_requests: None,
            scheduler_running_requests: Some(3),
            scheduler_waiting_requests: Some(2),
            kv_cache_usage: Some(0.125),
            prompt_tokens_total: Some(index * 100),
            generation_tokens_total: Some(index * 3),
            ttft_seconds: histogram.clone(),
            tpot_seconds: histogram.clone(),
            e2e_seconds: histogram,
        });
    }
    let stats = Arc::new(CachedStats(Mutex::new(history), window));
    let mut state_read_ns = Vec::new();
    for _ in 0..count {
        let start = Instant::now();
        black_box(
            stats
                .stats(&RouteTargetId::new("cpu-probe"), window)
                .unwrap(),
        );
        state_read_ns.push(start.elapsed().as_nanos());
    }

    let t1_path = output.with_extension("t1.jsonl");
    let (t1_sink, t1_writer) = RequestObservations::create(&t1_path).unwrap();
    let mut t1_pairs = Vec::new();
    for block in 0..=blocks {
        let on = router(Some(t1_sink.clone()), stats.clone());
        let off = router(None, stats.clone());
        let id = if block == 0 {
            "warmup".into()
        } else {
            format!("t1-{}", block - 1)
        };
        let (off_samples, on_samples) = if block % 2 == 1 {
            let off_samples = router_block(&off, &id, per_block, tokens.len()).await;
            let on_samples = router_block(&on, &id, per_block, tokens.len()).await;
            (off_samples, on_samples)
        } else {
            let on_samples = router_block(&on, &id, per_block, tokens.len()).await;
            let off_samples = router_block(&off, &id, per_block, tokens.len()).await;
            (off_samples, on_samples)
        };
        if block > 0 {
            t1_pairs.push(
                json!({"block":block-1,"order":if block%2 == 1 {"AB"} else {"BA"},
                "off":off_samples,"on":on_samples}),
            );
        }
    }
    let t1_counts = finish(t1_writer);
    let t1_rows: Vec<Value> = std::fs::read_to_string(&t1_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let mut feature_maintenance_ns = Vec::new();
    for row in &t1_rows {
        if let Some(id) = row["request_id"]
            .as_str()
            .filter(|id| id.starts_with("t1-"))
        {
            let index = id.rsplit('-').next().unwrap();
            let expected = if index == "cleanup" {
                0
            } else {
                index.parse::<usize>().unwrap() * tokens.len()
            };
            assert_eq!(
                row["pending_first_token_input_tokens"].as_u64().unwrap(),
                expected as u64
            );
            if index != "cleanup" {
                feature_maintenance_ns
                    .push(row["feature_maintenance_cost_ms"].as_f64().unwrap() * 1_000_000.0);
            }
        }
    }
    assert_eq!(feature_maintenance_ns.len(), count);

    let t2_path = output.with_extension("t2.jsonl");
    let (t2_sink, t2_writer) = RequestObservations::create(&t2_path).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let on = runtime(
        dir.path(),
        Some(t2_sink.clone()),
        stats.clone(),
        tokens.len(),
        deadline,
    );
    let off = runtime(dir.path(), None, stats.clone(), tokens.len(), deadline);
    let mut t2_pairs = Vec::new();
    for block in 0..=blocks {
        let id = if block == 0 {
            "warmup".into()
        } else {
            format!("t2-{}", block - 1)
        };
        let (off_samples, on_samples) = if block % 2 == 1 {
            let a = runtime_block(&off, &id, per_block, tokens.len()).await;
            let b = runtime_block(&on, &id, per_block, tokens.len()).await;
            (a, b)
        } else {
            let b = runtime_block(&on, &id, per_block, tokens.len()).await;
            let a = runtime_block(&off, &id, per_block, tokens.len()).await;
            (a, b)
        };
        if block > 0 {
            t2_pairs.push(
                json!({"block":block-1,"order":if block%2 == 1 {"AB"} else {"BA"},
                "off_ns":off_samples,"on_ns":on_samples}),
            );
        }
    }
    drop(on);
    drop(off);
    let t2_counts = finish(t2_writer);
    assert_eq!(
        t2_counts["written"].as_u64().unwrap(),
        (1 + 3 * (blocks + 1) * per_block) as u64
    );

    let mut labels = BTreeMap::<String, BTreeSet<String>>::new();
    for line in std::fs::read_to_string(&t2_path).unwrap().lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let event = row["event_type"].as_str().unwrap();
        if event == "clock_anchor" {
            continue;
        }
        let id = row["request_id"].as_str().unwrap().to_owned();
        assert!(labels.entry(id).or_default().insert(event.to_owned()));
        if event == "frontend_terminal" {
            assert_eq!(row["frontend_status"], "completed");
            assert_eq!(row["frontend_termination_reason"], "backend_length");
            assert_eq!(row["frontend_output_token_count"], 3);
            assert!(row["frontend_e2e_ms"].as_f64().unwrap() >= 0.0);
        }
        if event == "frontend_ttft" {
            assert_eq!(row["frontend_output_token_count"], 2);
            assert!(row["frontend_ttft_ms"].as_f64().unwrap() >= 0.0);
        }
    }
    assert_eq!(labels.len(), (blocks + 1) * per_block);
    let expected = BTreeSet::from([
        "router_selected".to_owned(),
        "frontend_ttft".to_owned(),
        "frontend_terminal".to_owned(),
    ]);
    assert!(labels.values().all(|events| *events == expected));

    let row = t1_rows
        .iter()
        .find(|row| row["event_type"] == "router_selected")
        .unwrap();
    let mut serialize_ns = Vec::new();
    for _ in 0..count {
        let start = Instant::now();
        black_box(serde_json::to_vec(black_box(row)).unwrap());
        serialize_ns.push(start.elapsed().as_nanos());
    }
    let writer_path = output.with_extension("writer.jsonl");
    let (sink, writer) = RequestObservations::create(&writer_path).unwrap();
    while sink.counts().written < 1 {
        assert_eq!(sink.counts().write_errors, 0);
        std::thread::yield_now();
    }
    let mut enqueue_ns = Vec::new();
    let mut write_ack_ns = Vec::new();
    for index in 0..count {
        let start = Instant::now();
        sink.record(black_box(row));
        enqueue_ns.push(start.elapsed().as_nanos());
        while sink.counts().written < (index + 2) as u64 {
            assert_eq!(sink.counts().write_errors, 0);
            std::thread::yield_now();
        }
        write_ack_ns.push(start.elapsed().as_nanos());
    }
    let writer_counts = finish(writer);
    serde_json::to_writer_pretty(file, &json!({
        "profile":"release", "debug_assertions":cfg!(debug_assertions),
        "parameters":{"paired_blocks":blocks,"per_block":per_block,"state_polls":state_polls,
            "state_window_ms":window.as_millis(),"request_deadline_ms":deadline.as_millis()},
        "tokenizer":{"backend":"fastokens","input_bytes":input.len(),"input_tokens":tokens.len()},
        "samples_ns":{"tokenizer_encode":tokenizer_ns,"cached_registry_state_read":state_read_ns,
            "router_feature_maintenance":feature_maintenance_ns,"writer_value_serialization":serialize_ns,
            "writer_value_serialization_enqueue":enqueue_ns,"writer_write_flush_ack":write_ack_ns},
        "t1_pairs":t1_pairs,"t1_t2_pairs":t2_pairs,
        "writer_counts":{"t1":t1_counts,"t1_t2":t2_counts,"writer_probe":writer_counts},
        "checks":{"same_route_and_output":true,"pending_load_and_cleanup":true},
    })).unwrap();
}
