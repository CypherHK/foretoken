// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Opt-in model-server EngineCore observations; serving never waits for file I/O.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde_json::{Value, json};
use zeromq::{
    DealerSocket, PullSocket, PushSocket, RouterSocket, Socket, SocketOptions, SocketRecv,
    SocketSend, ZmqMessage,
};

const OBSERVATION_PATH_ENV: &str = "FORETOKEN_MS_OBSERVATIONS_PATH";
const SCHEMA: &str = "stage2-ms-observation-v1";
const QUEUE_CAPACITY: usize = 4096;

/// Linux/Unix CLOCK_MONOTONIC, without loading an inference library or device.
fn clock_monotonic_ns() -> u64 {
    let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
    now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64
}

/// Writer loss counts; consumers cannot claim complete observations when either is nonzero.
#[derive(Default, Clone, Copy, serde::Serialize)]
pub struct Counts {
    pub written: u64,
    pub dropped: u64,
    pub write_errors: u64,
}

#[derive(Default)]
struct Counters {
    written: AtomicU64,
    dropped: AtomicU64,
    write_errors: AtomicU64,
}

impl Counters {
    fn snapshot(&self) -> Counts {
        Counts {
            written: self.written.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            write_errors: self.write_errors.load(Ordering::Relaxed),
        }
    }
}

enum Observation {
    Input(Vec<Bytes>, u64),
    Output(Vec<Bytes>, u64),
    Row(Value),
}

struct Inner {
    sender: Mutex<Option<mpsc::SyncSender<Observation>>>,
    counters: Arc<Counters>,
    relays: Mutex<Vec<tokio::task::JoinHandle<()>>>,
}

/// Cloneable nonblocking sink; the writer owner drains accepted messages at shutdown.
#[derive(Clone)]
pub struct EngineObservations(Arc<Inner>);

/// Owns the writer thread and relay tasks; no serving object owns its cleanup separately.
pub struct ObservationWriter {
    sink: EngineObservations,
    worker: Option<JoinHandle<()>>,
}

fn anchor(domain: &str) -> Value {
    let before = clock_monotonic_ns();
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("wall clock predates epoch")
        .as_nanos();
    let after = clock_monotonic_ns();
    json!({"schema_version":SCHEMA,"event_type":"ms_anchor","clock_domain":domain,
        "clock_monotonic_ns":before+(after-before)/2,"wall_unix_ns":wall,
        "sampling_span_ns":after-before,
        "boot_id":std::fs::read_to_string("/proc/sys/kernel/random/boot_id").ok().map(|x|x.trim().to_owned()),
        "time_ns_inode":std::fs::metadata("/proc/self/ns/time").ok().map(|m| {use std::os::unix::fs::MetadataExt;m.ino()})})
}

impl EngineObservations {
    /// Reads the sole opt-in switch; absent or empty means no writer, relay, or observation work.
    pub fn from_env(launch_config: Value) -> io::Result<Option<(Self, ObservationWriter)>> {
        match std::env::var_os(OBSERVATION_PATH_ENV).filter(|x| !x.is_empty()) {
            None => Ok(None),
            Some(path) => Self::create(Path::new(&path), launch_config, QUEUE_CAPACITY).map(Some),
        }
    }

    /// Creates an exclusive 0600 JSONL with a bounded nonblocking producer queue.
    fn create(
        path: &Path,
        launch_config: Value,
        capacity: usize,
    ) -> io::Result<(Self, ObservationWriter)> {
        use std::os::unix::fs::OpenOptionsExt;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)?;
        let counters = Arc::new(Counters::default());
        let domain = format!("ms-{}-{}", std::process::id(), clock_monotonic_ns());
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let sink = Self(Arc::new(Inner {
            sender: Mutex::new(Some(sender)),
            counters: counters.clone(),
            relays: Mutex::new(Vec::new()),
        }));
        let worker = std::thread::Builder::new().name("engine-observations".into()).spawn(move || {
            let mut writer = BufWriter::new(file);
            let mut state = DecodeState::default();
            let mut last_anchor = clock_monotonic_ns();
            write_row(&mut writer, &anchor(&domain), &counters);
            loop {
                let timeout = Duration::from_nanos(60_000_000_000_u64.saturating_sub(clock_monotonic_ns()-last_anchor));
                match receiver.recv_timeout(timeout) {
                    Ok(message) => {
                        for mut row in state.rows(message, &launch_config) {
                            row["schema_version"] = json!(SCHEMA); row["clock_domain"] = json!(domain);
                            write_row(&mut writer, &row, &counters);
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {},
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
                if clock_monotonic_ns()-last_anchor >= 60_000_000_000 {
                    write_row(&mut writer, &anchor(&domain), &counters); last_anchor = clock_monotonic_ns();
                }
            }
            // Emit explicit missing terminal records for requests that never produced an engine output.
            for (engine_id, request_id) in state.ids {
                if !state.finished.contains_key(&engine_id) {
                    write_row(&mut writer,&json!({"schema_version":SCHEMA,"event_type":"engine_request",
                        "clock_domain":domain,"request_id":request_id,"engine_request_id":engine_id,
                        "events":[],"prefill_stats":null,"events_receive_mono_ns":clock_monotonic_ns(),
                        "first_token_engine_mono_s":null,"first_token_time_source":null,
                        "terminal_status":"missing","missing_reason":"no_terminal_engine_output_before_writer_close"}),&counters);
                }
            }
            let counts = counters.snapshot();
            write_row(&mut writer,&json!({"schema_version":SCHEMA,"event_type":"ms_writer_tail",
                "clock_domain":domain,"receive_mono_ns":clock_monotonic_ns(),
                "written":counts.written,"dropped":counts.dropped,"write_errors":counts.write_errors}),&counters);
        })?;
        Ok((
            sink.clone(),
            ObservationWriter {
                sink,
                worker: Some(worker),
            },
        ))
    }

    fn enqueue(&self, row: Observation) {
        let sender = self.0.sender.lock().expect("observation queue poisoned");
        if sender.as_ref().is_none_or(|tx| tx.try_send(row).is_err()) {
            self.0.counters.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Records a whitelisted synthetic/test row without writing on the calling path.
    fn record(&self, row: Value) {
        self.enqueue(Observation::Row(row));
    }

    /// Returns current loss counters for run receipts; ownership stays with the writer.
    fn counts(&self) -> Counts {
        self.0.counters.snapshot()
    }

    /// Releases failed-attempt transport before an existing engine startup retry reuses its port.
    pub async fn stop_relays(&self) {
        let relays = std::mem::take(&mut *self.0.relays.lock().expect("relay owner poisoned"));
        for relay in &relays {
            relay.abort();
        }
        for relay in relays {
            let _ = relay.await;
        }
    }

    /// Inserts a local relay before the private client dispatcher strips engine fields.
    /// Only the audited single local EngineCore transport is supported by this opt-in path.
    pub async fn relay(&self, engine_handshake: &str) -> io::Result<String> {
        let mut handshake = RouterSocket::new();
        handshake
            .bind(engine_handshake)
            .await
            .map_err(io::Error::other)?;
        // Reserve a concrete endpoint; the existing client remains the handshake lifecycle owner.
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        let client_address = format!("tcp://{}", listener.local_addr()?);
        drop(listener);
        let address = client_address.clone();
        let sink = self.clone();
        let handle = tokio::spawn(async move {
            if let Err(error) = relay_transport(&mut handshake, &address, &sink).await {
                sink.record(
                    json!({"event_type":"ms_diagnostic","receive_mono_ns":clock_monotonic_ns(),
                    "missing_reason":"observation_relay_transport_failed"}),
                );
                tracing::error!(%error,"engine observation relay failed");
            }
        });
        self.0
            .relays
            .lock()
            .expect("relay owner poisoned")
            .push(handle);
        Ok(client_address)
    }
}

fn write_row(writer: &mut impl Write, row: &Value, counters: &Counters) {
    if serde_json::to_writer(&mut *writer, row)
        .and_then(|()| {
            writer
                .write_all(b"\n")
                .and_then(|()| writer.flush())
                .map_err(serde_json::Error::io)
        })
        .is_err()
    {
        counters.write_errors.fetch_add(1, Ordering::Relaxed);
    } else {
        counters.written.fetch_add(1, Ordering::Relaxed);
    }
}

impl ObservationWriter {
    fn close(&mut self) {
        for relay in self
            .sink
            .0
            .relays
            .lock()
            .expect("relay owner poisoned")
            .drain(..)
        {
            relay.abort();
        }
        self.sink
            .0
            .sender
            .lock()
            .expect("observation queue poisoned")
            .take();
        if let Some(worker) = self.worker.take()
            && worker.join().is_err()
        {
            self.sink
                .0
                .counters
                .write_errors
                .fetch_add(1, Ordering::Relaxed);
        }
    }
    /// Stops relays after engine drain, flushes the queue and records a final loss-count row.
    pub async fn finish(mut self) -> Counts {
        self.sink.stop_relays().await;
        self.close();
        self.sink.counts()
    }
}
impl Drop for ObservationWriter {
    fn drop(&mut self) {
        self.close();
    }
}

fn frames(parts: Vec<Bytes>) -> ZmqMessage {
    ZmqMessage::try_from(parts).expect("nonempty engine transport message")
}

/// Rewrites only local input/output addresses, retaining every other MsgPack key and value.
fn replace_init_addresses(
    metadata: &mut rmpv::Value,
    input: &str,
    output: &str,
) -> io::Result<(String, String)> {
    let rmpv::Value::Map(root) = metadata else {
        return Err(io::Error::other("INIT map missing"));
    };
    let Some((_, rmpv::Value::Map(addresses))) = root
        .iter_mut()
        .find(|(key, _)| key.as_str() == Some("addresses"))
    else {
        return Err(io::Error::other("INIT addresses map missing"));
    };
    if addresses
        .iter()
        .any(|(key, value)| key.as_str() == Some("coordinator_input") && !value.is_nil())
    {
        return Err(io::Error::other(
            "observation relay supports one local engine only",
        ));
    }
    let mut old = Vec::new();
    for (key, replacement) in [("inputs", input), ("outputs", output)] {
        let Some((_, rmpv::Value::Array(values))) = addresses
            .iter_mut()
            .find(|(name, _)| name.as_str() == Some(key))
        else {
            return Err(io::Error::other("INIT transport address missing"));
        };
        if values.len() != 1 {
            return Err(io::Error::other(
                "observation relay requires one local engine address",
            ));
        }
        old.push(
            values[0]
                .as_str()
                .ok_or_else(|| io::Error::other("INIT address is not a string"))?
                .to_owned(),
        );
        values[0] = replacement.into();
    }
    Ok((old.remove(0), old.remove(0)))
}

async fn relay_transport(
    handshake: &mut RouterSocket,
    address: &str,
    sink: &EngineObservations,
) -> io::Result<()> {
    let hello = handshake.recv().await.map_err(io::Error::other)?.into_vec();
    if hello.len() != 2 {
        return Err(io::Error::other("unexpected HELLO frames"));
    }
    let identity = hello[0].clone();
    let mut options = SocketOptions::default();
    options.peer_identity(identity.to_vec().try_into().map_err(io::Error::other)?);
    let mut upstream = DealerSocket::with_options(options);
    upstream.connect(address).await.map_err(io::Error::other)?;
    upstream
        .send(frames(hello[1..].to_vec()))
        .await
        .map_err(io::Error::other)?;
    let init = upstream.recv().await.map_err(io::Error::other)?.into_vec();
    if init.len() != 1 {
        return Err(io::Error::other("unexpected INIT frames"));
    }
    let mut metadata =
        rmpv::decode::read_value(&mut std::io::Cursor::new(&init[0])).map_err(io::Error::other)?;
    let mut input = RouterSocket::new();
    let input_address = input
        .bind("tcp://127.0.0.1:0")
        .await
        .map_err(io::Error::other)?
        .to_string();
    let mut output = PullSocket::new();
    let output_address = output
        .bind("tcp://127.0.0.1:0")
        .await
        .map_err(io::Error::other)?
        .to_string();
    let (original_input, original_output) =
        replace_init_addresses(&mut metadata, &input_address, &output_address)?;
    let mut input_options = SocketOptions::default();
    input_options.peer_identity(identity.to_vec().try_into().map_err(io::Error::other)?);
    let mut input_upstream = DealerSocket::with_options(input_options);
    input_upstream
        .connect(&original_input)
        .await
        .map_err(io::Error::other)?;
    let mut output_upstream = PushSocket::new();
    output_upstream
        .connect(&original_output)
        .await
        .map_err(io::Error::other)?;
    let mut payload = Vec::new();
    rmpv::encode::write_value(&mut payload, &metadata).map_err(io::Error::other)?;
    handshake
        .send(frames(vec![identity.clone(), payload.into()]))
        .await
        .map_err(io::Error::other)?;
    // Single owner forwards all frames byte-for-byte. Observation enqueue is bounded and nonblocking.
    loop {
        tokio::select! {
            ready=handshake.recv() => {let parts=ready.map_err(io::Error::other)?.into_vec();
                upstream.send(frames(parts[1..].to_vec())).await.map_err(io::Error::other)?;},
            registration=input.recv() => {let parts=registration.map_err(io::Error::other)?.into_vec();
                sink.enqueue(Observation::Input(parts[1..].to_vec(),clock_monotonic_ns()));
                input_upstream.send(frames(parts[1..].to_vec())).await.map_err(io::Error::other)?;},
            request=input_upstream.recv() => {let parts=request.map_err(io::Error::other)?.into_vec();
                sink.enqueue(Observation::Input(parts.clone(),clock_monotonic_ns()));
                let mut forwarded=vec![identity.clone()];forwarded.extend(parts);
                input.send(frames(forwarded)).await.map_err(io::Error::other)?;},
            response=output.recv() => {let parts=response.map_err(io::Error::other)?.into_vec();
                sink.enqueue(Observation::Output(parts.clone(),clock_monotonic_ns()));
                output_upstream.send(frames(parts)).await.map_err(io::Error::other)?;},
        }
    }
}

fn wire_json(bytes: &[u8]) -> Option<Value> {
    let value = rmpv::decode::read_value(&mut std::io::Cursor::new(bytes)).ok()?;
    serde_json::to_value(value).ok()
}
fn member(object: &Value, name: &str) -> Value {
    object.get(name).cloned().unwrap_or(Value::Null)
}

#[derive(Default)]
struct DecodeState {
    ids: BTreeMap<String, String>,
    finished: BTreeMap<String, bool>,
    first_token: BTreeMap<String, bool>,
    seen_events: BTreeMap<String, bool>,
    seq: u64,
}

impl DecodeState {
    fn rows(&mut self, message: Observation, config: &Value) -> Vec<Value> {
        match message {
            Observation::Row(row) => vec![row],
            Observation::Input(parts, receive) => {
                // Ready registration is a map; ADD request payload is array-like, including exact external_req_id.
                if parts.len() == 1 {
                    let Some(ready) = wire_json(&parts[0]) else {
                        return vec![
                            json!({"event_type":"ms_diagnostic","receive_mono_ns":receive,"missing_reason":"ready_decode_failed"}),
                        ];
                    };
                    if ready.get("max_model_len").is_some() {
                        return vec![json!({"event_type":"engine_ready",
                        "max_model_len":member(&ready,"max_model_len"),"num_gpu_blocks":member(&ready,"num_gpu_blocks"),
                        "receive_mono_ns":receive,"config":config})];
                    }
                } else if parts[0].as_ref() == [0]
                    && let Some(request) = wire_json(&parts[1])
                    && let Some(id) = request.get(0).and_then(Value::as_str)
                {
                    let external = request.get(15).and_then(Value::as_str);
                    if let Some(external) = external {
                        self.ids.insert(id.to_owned(), external.to_owned());
                    }
                }
                vec![]
            }
            Observation::Output(parts, receive) => {
                let Some(batch) = parts.first().and_then(|part| wire_json(part)) else {
                    return vec![
                        json!({"event_type":"ms_diagnostic","receive_mono_ns":receive,"missing_reason":"output_decode_failed"}),
                    ];
                };
                let Some(batch) = batch.as_array() else {
                    return vec![];
                };
                let mut rows = Vec::new();
                let engine_time = batch.get(3).cloned().unwrap_or(Value::Null);
                if let Some(stats) = batch.get(2).filter(|x| x.is_object()) {
                    self.seq += 1;
                    let prefix = stats.get("prefix_cache_stats").unwrap_or(&Value::Null);
                    rows.push(json!({"event_type":"scheduler_stats","engine_index":batch.first(),
                        "envelope_engine_mono_s":engine_time,"receive_mono_ns":receive,"receiver_seq":self.seq,
                        "num_running_reqs":member(stats,"num_running_reqs"),"num_waiting_reqs":member(stats,"num_waiting_reqs"),
                        "num_skipped_waiting_reqs":member(stats,"num_skipped_waiting_reqs"),"kv_cache_usage":member(stats,"kv_cache_usage"),
                        "prefix_cache":{"queries":member(prefix,"queries"),"hits":member(prefix,"hits"),
                            "preempted_requests":member(prefix,"preempted_requests"),"preempted_queries":member(prefix,"preempted_queries"),"preempted_hits":member(prefix,"preempted_hits")}}));
                }
                for output in batch.get(1).and_then(Value::as_array).into_iter().flatten() {
                    let Some(output) = output.as_array() else {
                        continue;
                    };
                    let Some(id) = output.first().and_then(Value::as_str) else {
                        continue;
                    };
                    let events=output.get(7).and_then(Value::as_array).map(|events|events.iter().map(|event|{
                        let kind=event.get("type").and_then(Value::as_u64);
                        json!({"type":match kind {Some(1)=>"QUEUED",Some(2)=>"SCHEDULED",Some(3)=>"PREEMPTED",_=>"unknown"},"engine_mono_s":member(event,"timestamp")})
                    }).collect::<Vec<_>>()).unwrap_or_default();
                    // v0.20 prefill is at 10. Retain only audited fields; absence stays null, never default zero.
                    let stats = output.get(10).filter(|x| x.is_object());
                    let prefill=stats.map(|stats|json!({"num_prompt_tokens":member(stats,"num_prompt_tokens"),"num_computed_tokens":member(stats,"num_computed_tokens"),
                        "num_cached_tokens":member(stats,"num_cached_tokens"),"num_local_cached_tokens":member(stats,"num_local_cached_tokens"),
                        "num_external_cached_tokens":member(stats,"num_external_cached_tokens"),"num_cache_creation_tokens":member(stats,"num_cache_creation_tokens")}));
                    let has_tokens = output
                        .get(1)
                        .and_then(Value::as_array)
                        .is_some_and(|x| !x.is_empty());
                    let first = has_tokens && !self.first_token.contains_key(id);
                    if has_tokens {
                        self.first_token.insert(id.to_owned(), true);
                    }
                    if !events.is_empty() {
                        self.seen_events.insert(id.to_owned(), true);
                    }
                    let terminal = output.get(5).and_then(Value::as_u64);
                    if terminal.is_some() {
                        self.finished.insert(id.to_owned(), true);
                    }
                    if !events.is_empty() || stats.is_some() || first || terminal.is_some() {
                        let external = self.ids.get(id);
                        let missing = if external.is_none() {
                            Some("external_request_id_mapping_missing")
                        } else if !self.seen_events.contains_key(id) {
                            Some("engine_events_not_received")
                        } else {
                            None
                        };
                        rows.push(json!({"event_type":"engine_request","request_id":external,"engine_request_id":id,
                            "events":events,"prefill_stats":prefill,"events_receive_mono_ns":receive,
                            "first_token_engine_mono_s":if first{engine_time.clone()}else{Value::Null},
                            "first_token_time_source":if first{Some("first_token_output_envelope")}else{None},
                            "terminal_status":match terminal{Some(0)=>"stop",Some(1)=>"length",Some(2)=>"abort",Some(3)=>"error",Some(4)=>"repetition",_=>"pending"},
                            "missing_reason":missing}));
                    }
                }
                rows
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, GenerateInput, VllmBackend};
    use futures::StreamExt;
    use vllm_engine_core_client::mock_engine::{MockEngineConfig, connect_to_frontend};
    use vllm_engine_core_client::{EngineCoreClient, EngineCoreClientConfig, EngineCoreProtocol};
    use vllm_llm::Llm;

    fn private_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "ms-observation-{name}-{}.jsonl",
            uuid::Uuid::new_v4()
        ))
    }
    fn bytes(value: &Value) -> Bytes {
        rmp_serde::to_vec_named(value).unwrap().into()
    }
    fn event(kind: u64, time: f64) -> Value {
        json!({"type":kind,"timestamp":time})
    }
    fn output(id: &str, events: Value, stats: Value, terminal: Value) -> Value {
        json!([
            id,
            [7],
            null,
            null,
            null,
            terminal,
            null,
            events,
            null,
            null,
            stats,
            null,
            0
        ])
    }

    // A failed persistent-cache attempt must release its observation socket before the existing retry.
    #[tokio::test]
    async fn failed_attempt_releases_handshake_address_before_retry() {
        let path = private_path("retry");
        let (sink, writer) = EngineObservations::create(&path, json!({}), 16).unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("tcp://{}", listener.local_addr().unwrap());
        drop(listener);
        sink.relay(&address).await.unwrap();
        sink.stop_relays().await;
        // The same listener can be bound again even though the writer is kept for the next attempt.
        sink.relay(&address).await.unwrap();
        let counts = writer.finish().await;
        assert_eq!(counts.dropped, 0);
        assert_eq!(counts.write_errors, 0);
        std::fs::remove_file(path).unwrap();
    }

    // A bounded producer may lose observations under load, never silently or by blocking request work.
    #[tokio::test]
    async fn queue_loss_and_exclusive_file_are_observable() {
        let path = private_path("loss");
        let (sink, writer) = EngineObservations::create(&path, json!({}), 1).unwrap();
        assert!(EngineObservations::create(&path, json!({}), 1).is_err());
        for _ in 0..10000 {
            sink.record(json!({"event_type":"ms_diagnostic","receive_mono_ns":clock_monotonic_ns(),"missing_reason":"fixture"}));
        }
        let counts = writer.finish().await;
        let rows: Vec<Value> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|x| serde_json::from_str(x).unwrap())
            .collect();
        assert_eq!(rows[0]["event_type"], "ms_anchor");
        assert_eq!(rows.last().unwrap()["event_type"], "ms_writer_tail");
        assert_eq!(rows.last().unwrap()["dropped"], counts.dropped);
        assert!(counts.dropped > 0);
        assert_eq!(counts.write_errors, 0);
        std::fs::remove_file(path).unwrap();
    }

    // Engine output may arrive out of engine-time order after preemption; keep every event and null absent fields.
    #[test]
    fn all_reschedules_and_missing_prefill_are_preserved_in_receive_order() {
        let mut state = DecodeState::default();
        state.ids.insert("engine".into(), "cmpl-fixture".into());
        let prefill = json!({"num_prompt_tokens":10,"num_computed_tokens":6,"num_cached_tokens":4,"num_local_cached_tokens":4,"num_external_cached_tokens":0});
        let first = json!([
            0,
            [output(
                "engine",
                json!([event(1, 1.), event(2, 2.), event(3, 2.5), event(2, 3.)]),
                prefill,
                json!(null)
            )],
            null,
            3.1
        ]);
        let rows = state.rows(Observation::Output(vec![bytes(&first)], 100), &json!({}));
        assert_eq!(rows[0]["events"].as_array().unwrap().len(), 4);
        assert_eq!(
            rows[0]["prefill_stats"]["num_cache_creation_tokens"],
            Value::Null
        );
        assert_eq!(rows[0]["request_id"], "cmpl-fixture");
        let last = json!([
            0,
            [output(
                "engine",
                json!([event(3, 2.7), event(2, 2.8)]),
                Value::Null,
                json!(1)
            )],
            null,
            3.
        ]);
        let rows = state.rows(Observation::Output(vec![bytes(&last)], 101), &json!({}));
        assert_eq!(rows[0]["events"][0]["engine_mono_s"], 2.7);
        assert_eq!(rows[0]["events_receive_mono_ns"], 101);
        assert_eq!(rows[0]["prefill_stats"], Value::Null);
        assert_eq!(rows[0]["terminal_status"], "length");
        state
            .ids
            .insert("no-events".into(), "cmpl-no-events".into());
        let no_events = json!([
            0,
            [output("no-events", Value::Null, Value::Null, json!(1))],
            null,
            4.
        ]);
        let none = state.rows(
            Observation::Output(vec![bytes(&no_events)], 102),
            &json!({}),
        );
        assert_eq!(none[0]["request_id"], "cmpl-no-events");
        assert_eq!(none[0]["missing_reason"], "engine_events_not_received");
        assert_eq!(rows[0]["missing_reason"], Value::Null);
        let missing = json!([
            0,
            [output("unmapped", Value::Null, Value::Null, json!(2))],
            null,
            4.
        ]);
        assert_eq!(
            state.rows(Observation::Output(vec![bytes(&missing)], 102), &json!({}))[0]["missing_reason"],
            "external_request_id_mapping_missing"
        );
    }

    // Typed reserialization used to erase unknown INIT extensions; only the two endpoint values may change.
    #[test]
    fn init_address_rewrite_preserves_unknown_msgpack_extensions() {
        use rmpv::Value as M;
        let addresses = M::Map(vec![
            ("inputs".into(), M::Array(vec!["tcp://old-input".into()])),
            ("outputs".into(), M::Array(vec!["tcp://old-output".into()])),
            ("coordinator_input".into(), M::Nil),
            ("opaque_address_field".into(), M::Binary(vec![0, 255])),
        ]);
        let mut init = M::Map(vec![
            ("addresses".into(), addresses),
            (
                "parallel_config".into(),
                M::Map(vec![("unknown".into(), M::Ext(7, vec![1, 2, 3]))]),
            ),
            ("unknown_init_extension".into(), M::Ext(4, vec![9, 8, 7])),
        ]);
        let original = init.clone();
        assert_eq!(
            replace_init_addresses(&mut init, "tcp://new-input", "tcp://new-output").unwrap(),
            ("tcp://old-input".into(), "tcp://old-output".into())
        );
        let mut payload = Vec::new();
        rmpv::encode::write_value(&mut payload, &init).unwrap();
        let mut decoded = rmpv::decode::read_value(&mut std::io::Cursor::new(payload)).unwrap();
        replace_init_addresses(&mut decoded, "tcp://old-input", "tcp://old-output").unwrap();
        assert_eq!(decoded, original);
    }

    async fn fixture(on: bool, aborted: bool) -> (Vec<Value>, Vec<Value>) {
        let path = private_path("relay");
        let observations = on.then(|| {
            EngineObservations::create(
                &path,
                json!({"block_size":{"value":16,"source":"config"},"max_num_seqs":{"value":16,"source":"config"},"max_num_batched_tokens":{"value":2048,"source":"config"},"dtype":{"value":"bfloat16","source":"config"}}),
                4096,
            )
            .unwrap()
        });
        let port = vllm_managed_engine::allocate_handshake_port("127.0.0.1").unwrap();
        let engine_address = format!("tcp://127.0.0.1:{port}");
        let client_address = if let Some((sink, _)) = &observations {
            sink.relay(&engine_address).await.unwrap()
        } else {
            engine_address.clone()
        };
        let engine = tokio::spawn(async move {
            let mut engine =
                connect_to_frontend(&engine_address, &[0, 0], MockEngineConfig::default())
                    .await
                    .unwrap();
            let mut data = engine.data_sockets.remove(0);
            let request_frames = data.dealer.recv().await.unwrap().into_vec();
            assert_eq!(request_frames[0].as_ref(), [0]);
            let request = wire_json(&request_frames[1]).unwrap();
            let id = request[0].as_str().unwrap();
            assert!(id.starts_with("cmpl-fixture-"));
            if aborted {
                let abort = data.dealer.recv().await.unwrap().into_vec();
                assert_eq!(abort[0].as_ref(), [1]);
                assert_eq!(wire_json(&abort[1]).unwrap(), json!([id]));
            }
            let prefill = json!({"num_prompt_tokens":1,"num_computed_tokens":1,"num_cached_tokens":0,"num_local_cached_tokens":0,"num_external_cached_tokens":0});
            let stats = json!({"num_running_reqs":1,"num_waiting_reqs":0,"num_skipped_waiting_reqs":0,"step_counter":0,"current_wave":0,"kv_cache_eviction_events":[],"kv_cache_usage":0.,
                "prefix_cache_stats":{"reset":false,"requests":1,"queries":1,"hits":0,"preempted_requests":0,"preempted_queries":0,"preempted_hits":0}});
            let batch = json!([
                0,
                [output(
                    id,
                    json!([event(1, 1.), event(2, 2.), event(3, 2.01), event(2, 2.05)]),
                    prefill,
                    json!(if aborted { 2 } else { 1 })
                )],
                stats,
                2.1
            ]);
            data.push.send(frames(vec![bytes(&batch)])).await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        });
        let client = EngineCoreClient::connect_with_protocol(
            EngineCoreClientConfig::new_single(&client_address).with_model_name("fixture"),
            EngineCoreProtocol::V0_20,
        )
        .await
        .unwrap();
        let backend = VllmBackend::new(Llm::new(client), None);
        let request = GenerateInput {
            request_id: "cmpl-fixture".into(),
            prompt_token_ids: vec![1],
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
        };
        let stream = backend.generate(request).await.unwrap();
        if aborted {
            backend.abort(&["cmpl-fixture".into()]).await.unwrap();
        }
        let outputs: Vec<Value> = stream
            .map(|item| serde_json::to_value(item.unwrap()).unwrap())
            .collect()
            .await;
        backend.shutdown().await.unwrap();
        engine.await.unwrap();
        let rows = if let Some((_, writer)) = observations {
            assert_eq!(writer.finish().await.dropped, 0);
            if let Some(directory) = std::env::var_os("FORETOKEN_T11_FIXTURE_ARTIFACTS") {
                let directory = std::path::PathBuf::from(directory);
                std::fs::create_dir_all(&directory).unwrap();
                let name = if aborted {
                    "abort-events.jsonl"
                } else {
                    "normal-events.jsonl"
                };
                let destination = directory.join(name);
                assert!(!destination.exists());
                std::fs::copy(&path, destination).unwrap();
            }
            let rows = std::fs::read_to_string(&path)
                .unwrap()
                .lines()
                .map(|x| serde_json::from_str(x).unwrap())
                .collect();
            std::fs::remove_file(path).unwrap();
            rows
        } else {
            vec![]
        };
        (outputs, rows)
    }

    // The opt-in relay must preserve the actual Llm/HTTP token path, randomized engine identity and metadata.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn observation_switch_preserves_llm_output_and_request_identity() {
        vllm_tracing::init_tracing("T11Cpu");
        for aborted in [false, true] {
            let off = tokio::time::timeout(Duration::from_secs(10), fixture(false, aborted))
                .await
                .unwrap();
            let on = tokio::time::timeout(Duration::from_secs(10), fixture(true, aborted))
                .await
                .unwrap();
            assert_eq!(off.0, on.0);
            assert!(off.1.is_empty());
            let request =
                on.1.iter()
                    .find(|r| r["event_type"] == "engine_request")
                    .unwrap();
            assert_eq!(request["request_id"], "cmpl-fixture");
            assert!(
                request["engine_request_id"]
                    .as_str()
                    .unwrap()
                    .starts_with("cmpl-fixture-")
            );
            assert_eq!(request["events"].as_array().unwrap().len(), 4);
            assert!(on.1.iter().any(|r| r["event_type"] == "engine_ready"));
            assert!(on.1.iter().any(|r| r["event_type"] == "scheduler_stats"));
        }
    }
    // Opt-in release-only enqueue probe separates serving-path cost from background decode/flush and drain.
    #[tokio::test]
    async fn release_observation_enqueue_overhead_probe() {
        let Some(directory) = std::env::var_os("FORETOKEN_T11_CPU_PROBE_DIR") else {
            return;
        };
        assert!(
            !std::hint::black_box(cfg!(debug_assertions)),
            "CPU overhead measurement requires release profile"
        );
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("observations.jsonl");
        let (sink, writer) = EngineObservations::create(&path, json!({}), 4096).unwrap();
        let stats = json!({"num_running_reqs":12,"num_waiting_reqs":3,"num_skipped_waiting_reqs":1,"kv_cache_usage":0.3,
            "prefix_cache_stats":{"queries":40,"hits":20,"preempted_requests":0,"preempted_queries":0,"preempted_hits":0}});
        let packet = vec![bytes(&json!([
            0,
            [],
            stats,
            clock_monotonic_ns() as f64 / 1e9
        ]))];
        let mut raw = Vec::new();
        for block in 0..10 {
            for index in 0..110 {
                let mut pair = [0_u128; 2];
                for on in if block % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    let started = std::time::Instant::now();
                    if on {
                        sink.enqueue(Observation::Output(packet.clone(), clock_monotonic_ns()));
                    } else {
                        std::hint::black_box(&packet);
                    }
                    pair[usize::from(on)] = started.elapsed().as_nanos();
                }
                if index >= 10 {
                    raw.push(json!({"block":block,"off_ns":pair[0],"on_ns":pair[1],"on_minus_off_ns":pair[1] as i128-pair[0] as i128}));
                }
            }
        }
        let counts = writer.finish().await;
        assert_eq!(counts.dropped, 0);
        assert_eq!(counts.write_errors, 0);
        let result = json!({"scope":"release CPU bounded raw-output enqueue; background decoder/flush and drain excluded",
            "samples":raw,"counts":counts,"blocks":10,"warmup_pairs_per_block":10,"measured_pairs_per_block":100,
            "timer":"Instant","paired_order":"alternating OFF/ON then ON/OFF by block",
            "packet_bytes":packet.iter().map(Bytes::len).sum::<usize>()});
        std::fs::write(
            directory.join("samples.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
    }
    // The real writer must refresh its anchor even when no requests arrive; this opt-in probe waits one period.
    #[tokio::test]
    async fn periodic_anchor_cpu_probe() {
        let Some(directory) = std::env::var_os("FORETOKEN_T11_ANCHOR_PROBE_DIR") else {
            return;
        };
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("anchors.jsonl");
        let (_, writer) = EngineObservations::create(&path, json!({}), 4096).unwrap();
        tokio::time::sleep(Duration::from_millis(61100)).await;
        assert_eq!(writer.finish().await.dropped, 0);
        let rows: Vec<Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        let anchors: Vec<_> = rows
            .iter()
            .filter(|r| r["event_type"] == "ms_anchor")
            .collect();
        assert_eq!(anchors.len(), 2);
        let interval = anchors[1]["clock_monotonic_ns"].as_u64().unwrap()
            - anchors[0]["clock_monotonic_ns"].as_u64().unwrap();
        assert!((60_000_000_000..62_000_000_000).contains(&interval));
    }
    // One local mock keeps EngineCore, Llm and backend lifecycles real while excluding model compute and HTTP.
    async fn roundtrip_block(on: bool, path: &Path, total: usize) -> (Vec<u128>, Option<Counts>) {
        let config = json!({"block_size":{"value":16,"source":"config"},"max_num_seqs":{"value":16,"source":"config"},
            "max_num_batched_tokens":{"value":2048,"source":"config"},"dtype":{"value":"bfloat16","source":"config"}});
        let observed = on.then(|| EngineObservations::create(path, config, 4096).unwrap());
        let port = vllm_managed_engine::allocate_handshake_port("127.0.0.1").unwrap();
        let engine_address = format!("tcp://127.0.0.1:{port}");
        let client_address = if let Some((sink, _)) = &observed {
            sink.relay(&engine_address).await.unwrap()
        } else {
            engine_address.clone()
        };
        let engine = tokio::spawn(async move {
            let mut engine =
                connect_to_frontend(&engine_address, &[0, 0], MockEngineConfig::default())
                    .await
                    .unwrap();
            let mut data = engine.data_sockets.remove(0);
            for _ in 0..total {
                let request_frames = data.dealer.recv().await.unwrap().into_vec();
                assert_eq!(request_frames[0].as_ref(), [0]);
                let request = wire_json(&request_frames[1]).unwrap();
                let id = request[0].as_str().unwrap();
                let stamp = clock_monotonic_ns() as f64 / 1e9;
                let prefill = json!({"num_prompt_tokens":1,"num_computed_tokens":1,"num_cached_tokens":0,"num_local_cached_tokens":0,"num_external_cached_tokens":0});
                let stats = json!({"num_running_reqs":1,"num_waiting_reqs":0,"num_skipped_waiting_reqs":0,"step_counter":0,
                    "current_wave":0,"kv_cache_eviction_events":[],"kv_cache_usage":0.,
                    "prefix_cache_stats":{"reset":false,"requests":1,"queries":1,"hits":0,"preempted_requests":0,"preempted_queries":0,"preempted_hits":0}});
                let batch = json!([
                    0,
                    [output(
                        id,
                        json!([event(1, stamp - 0.001), event(2, stamp - 0.0005)]),
                        prefill,
                        json!(1)
                    )],
                    stats,
                    stamp
                ]);
                data.push.send(frames(vec![bytes(&batch)])).await.unwrap();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        });
        let client = EngineCoreClient::connect_with_protocol(
            EngineCoreClientConfig::new_single(&client_address).with_model_name("roundtrip"),
            EngineCoreProtocol::V0_20,
        )
        .await
        .unwrap();
        let backend = VllmBackend::new(Llm::new(client), None);
        let mut durations = Vec::new();
        for i in 0..total {
            let id = format!("cmpl-roundtrip-{i}");
            let request = GenerateInput {
                request_id: id.clone(),
                prompt_token_ids: vec![1],
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
            };
            let start = std::time::Instant::now();
            let mut stream = backend.generate(request).await.unwrap();
            let mut count = 0;
            while let Some(item) = stream.next().await {
                let crate::backend::TokenEvent::Token(token) = item.unwrap() else {
                    panic!("unexpected mock error")
                };
                assert_eq!(token.request_id, id);
                assert_eq!(token.token_ids, vec![7]);
                count += 1;
            }
            let elapsed = start.elapsed().as_nanos();
            assert_eq!(count, 1);
            durations.push(elapsed);
            assert_eq!(backend.telemetry().running_requests, 0);
        }
        backend.shutdown().await.unwrap();
        engine.await.unwrap();
        let counts = if let Some((_, writer)) = observed {
            let c = writer.finish().await;
            assert_eq!(c.dropped, 0);
            assert_eq!(c.write_errors, 0);
            Some(c)
        } else {
            None
        };
        (durations, counts)
    }

    // Paired wall time includes the opt-in socket hop and asynchronous writer interference, not only enqueue cost.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn release_model_server_roundtrip_overhead_probe() {
        let Some(directory) = std::env::var_os("FORETOKEN_T11_ROUNDTRIP_PROBE_DIR") else {
            return;
        };
        assert!(
            !std::hint::black_box(cfg!(debug_assertions)),
            "roundtrip CPU probe requires release"
        );
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir(&directory).unwrap();
        let mut raw = Vec::new();
        let mut loss = Vec::new();
        for block in 0..8 {
            let mut pairs = [Vec::new(), Vec::new()];
            for on in if block % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let path = directory.join(format!("block-{block}-events.jsonl"));
                let (values, counts) =
                    tokio::time::timeout(Duration::from_secs(20), roundtrip_block(on, &path, 60))
                        .await
                        .unwrap();
                pairs[usize::from(on)] = values;
                if let Some(counts) = counts {
                    loss.push(counts);
                }
            }
            for (i, (off, on)) in pairs[0].iter().zip(&pairs[1]).enumerate().skip(10) {
                raw.push(json!({"block":block,"index":i-10,"off_ns":off,"on_ns":on,
                "on_minus_off_ns":*on as i128-*off as i128}));
            }
        }
        let result = json!({"scope":"release localhost mock EngineCore→real client/Llm/backend roundtrip wall time; ON includes relay and background decoder/flush interference; excludes engine computation, HTTP/frontend and drain",
            "samples":raw,"blocks":8,"warmup_requests_per_mode_per_block":10,"paired_requests_per_block":50,
            "request_contract":"one prompt token, one output token, one terminal; output+external ID and inflight cleanup asserted every request",
            "paired_order":"OFF/ON then ON/OFF alternating by block","writer_counts":loss});
        std::fs::write(
            directory.join("samples.json"),
            serde_json::to_vec_pretty(&result).unwrap(),
        )
        .unwrap();
    }
}
