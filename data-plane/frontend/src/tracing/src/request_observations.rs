// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Opt-in local request observations with one process clock and a bounded writer.

use std::fs::OpenOptions;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

const QUEUE_CAPACITY: usize = 4096;

#[derive(Default)]
struct Counters {
    written: AtomicU64,
    dropped: AtomicU64,
    write_errors: AtomicU64,
}

struct Inner {
    origin: Instant,
    clock_domain: String,
    sender: Mutex<Option<mpsc::SyncSender<Vec<u8>>>>,
    counters: Arc<Counters>,
}

/// Shared optional channel for typed, whitelisted observations. No file I/O occurs in `record`.
#[derive(Clone)]
pub struct RequestObservations(Arc<Inner>);

/// Writer counters; a nonzero loss/error count invalidates completeness claims.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ObservationCounts {
    pub written: u64,
    pub dropped: u64,
    pub write_errors: u64,
}

/// Owns the writer thread; dropping it closes the channel, drains accepted rows and joins it.
pub struct ObservationWriter {
    inner: Arc<Inner>,
    worker: Option<JoinHandle<()>>,
}

impl RequestObservations {
    /// Creates a new private JSONL file, refusing to overwrite an existing path.
    /// Retain the returned writer until all observed requests have drained.
    pub fn create(path: impl AsRef<Path>) -> io::Result<(Self, ObservationWriter)> {
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(path)?;
        let (sender, receiver) = mpsc::sync_channel::<Vec<u8>>(QUEUE_CAPACITY);
        let counters = Arc::new(Counters::default());
        let inner = Arc::new(Inner {
            origin: Instant::now(),
            // Wall time is an instance identity only, never a timing conversion.
            clock_domain: format!(
                "frontend-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            sender: Mutex::new(Some(sender)),
            counters: counters.clone(),
        });
        let worker = std::thread::Builder::new()
            .name("request-observations".into())
            .spawn(move || {
                let mut writer = BufWriter::new(file);
                let mut failed = false;
                for row in receiver {
                    if failed {
                        counters.write_errors.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    // Flush each row so first-token labels can be consumed before completion.
                    if writer
                        .write_all(&row)
                        .and_then(|()| writer.write_all(b"\n"))
                        .and_then(|()| writer.flush())
                        .is_err()
                    {
                        failed = true;
                        counters.write_errors.fetch_add(1, Ordering::Relaxed);
                        tracing::error!(
                            "request observation writer failed; subsequent rows are counted as lost"
                        );
                    } else {
                        counters.written.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })?;
        Ok((
            Self(inner.clone()),
            ObservationWriter {
                inner,
                worker: Some(worker),
            },
        ))
    }

    /// Identifies this frontend's monotonic clock; unrelated processes cannot compare these offsets.
    pub fn clock_domain(&self) -> &str {
        &self.0.clock_domain
    }

    /// Converts a timestamp from this process; earlier-than-origin times are explicitly unknown.
    pub fn offset_ms(&self, at: Instant) -> Option<f64> {
        at.checked_duration_since(self.0.origin)
            .map(|d| d.as_secs_f64() * 1000.0)
    }

    /// Serializes and attempts a nonblocking enqueue. Call only outside routing locks.
    /// Queue saturation or shutdown never blocks serving work and is counted, not silently discarded.
    pub fn record(&self, row: &impl Serialize) {
        let Ok(bytes) = serde_json::to_vec(row) else {
            self.0.counters.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let sender = self
            .0
            .sender
            .lock()
            .expect("observation channel lock poisoned");
        if sender
            .as_ref()
            .is_none_or(|sender| sender.try_send(bytes).is_err())
        {
            self.0.counters.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Returns the current cumulative writer counters without exposing recorded content.
    pub fn counts(&self) -> ObservationCounts {
        ObservationCounts {
            written: self.0.counters.written.load(Ordering::Relaxed),
            dropped: self.0.counters.dropped.load(Ordering::Relaxed),
            write_errors: self.0.counters.write_errors.load(Ordering::Relaxed),
        }
    }
}

impl ObservationWriter {
    fn close(&mut self) {
        self.inner
            .sender
            .lock()
            .expect("observation channel lock poisoned")
            .take();
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                self.inner
                    .counters
                    .write_errors
                    .fetch_add(1, Ordering::Relaxed);
            }
            tracing::info!(
                written = self.inner.counters.written.load(Ordering::Relaxed),
                dropped = self.inner.counters.dropped.load(Ordering::Relaxed),
                write_errors = self.inner.counters.write_errors.load(Ordering::Relaxed),
                "request observation writer drained"
            );
        }
    }

    /// Drains all accepted rows and returns counters for fixture/completeness verification.
    pub fn finish(mut self) -> ObservationCounts {
        self.close();
        RequestObservations(self.inner.clone()).counts()
    }
}

impl Drop for ObservationWriter {
    fn drop(&mut self) {
        self.close();
    }
}
