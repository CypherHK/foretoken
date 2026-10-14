// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the Foretoken project

//! Private experiment on the main API's accepted sockets; failures use runtime shutdown.

use std::io;
use std::net::SocketAddr;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

const ENV: &str = "FORETOKEN_EXPERIMENT_API_SOCKET";

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Disabled,
    Default,
    Nodelay,
}

impl Mode {
    /// Parse before engine startup; unset keeps the existing socket behavior.
    pub(super) fn from_env() -> io::Result<Self> {
        match std::env::var(ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::Disabled),
            Ok(value) if value == "default" => Ok(Self::Default),
            Ok(value) if value == "nodelay" => Ok(Self::Nodelay),
            _ => Err(io::Error::new(io::ErrorKind::InvalidInput, format!("invalid {ENV}"))),
        }
    }
}

pub(super) struct Listener {
    inner: TcpListener,
    mode: Mode,
    accepted: u64,
    failed: UnboundedSender<io::Error>,
}

impl Listener {
    /// Wrap only the main API listener, sharing one readback/count path for both arms.
    pub(super) fn new(inner: TcpListener, mode: Mode) -> (Self, UnboundedReceiver<io::Error>) {
        let (failed, receiver) = unbounded_channel();
        (Self { inner, mode, accepted: 0, failed }, receiver)
    }

    fn observe(&mut self, stream: &TcpStream) -> io::Result<()> {
        if matches!(self.mode, Mode::Disabled) {
            return Ok(());
        }
        if matches!(self.mode, Mode::Nodelay) {
            stream.set_nodelay(true)?;
        }
        let actual = stream.nodelay()?;
        if matches!(self.mode, Mode::Nodelay) && !actual {
            return Err(io::Error::other("TCP_NODELAY readback false in nodelay arm"));
        }
        self.accepted = self.accepted.checked_add(1)
            .ok_or_else(|| io::Error::other("API experiment accepted counter overflow"))?;
        tracing::info!(accepted = self.accepted, nodelay = actual,
            arm = if matches!(self.mode, Mode::Nodelay) { "nodelay" } else { "default" },
            "API socket experiment accepted");
        Ok(())
    }
}

impl axum::serve::Listener for Listener {
    type Io = TcpStream;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        // Reuse axum's ordinary TCP accept/error handling without changing defaults.
        let (stream, address) = axum::serve::Listener::accept(&mut self.inner).await;
        if let Err(error) = self.observe(&stream) {
            // Listener::accept cannot return errors; notify runtime, drop this socket,
            // then await the existing owner shutdown instead of accepting more work.
            let _ = self.failed.send(error);
            drop(stream);
            return std::future::pending().await;
        }
        (stream, address)
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}
