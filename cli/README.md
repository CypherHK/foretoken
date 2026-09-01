<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
-->

# Foretoken CLI

English | [简体中文](README_zh.md)

The Foretoken CLI installs the platform, deploys Kustomize configurations, reports serving readiness, resolves frontend endpoints, and exposes benchmarks through one `foretoken` entry point.

Install the Foretoken CLI with pip or uv:

```bash
pip install -e .
# or
uv pip install -e .
```

Install or upgrade the platform in local access mode:

```bash
foretoken install
```

The command reuses one compatible Prometheus instance or installs a CLI-managed monitoring stack when none exists. On NVIDIA GPU nodes it reuses one ready DCGM Exporter with a working ServiceMonitor, or installs a CLI-managed exporter when none exists. A shared Prometheus must select that ServiceMonitor. Duplicate, unhealthy, incomplete, or unmonitored exporters stop installation instead of being overlaid. On MetaX GPU nodes the CLI requires and reuses one platform-managed mxExporter with a working ServiceMonitor. The CLI never installs GPU drivers, device plugins, or vendor operators. Use `--prometheus NAMESPACE/NAME` only when automatic discovery finds multiple compatible instances.

Gateway mode can create a dedicated Gateway or reuse an existing one:

```bash
foretoken install --frontend-mode gateway --create-gateway

foretoken install \
  --frontend-mode gateway \
  --gateway-name inference-gateway \
  --gateway-namespace gateway-system \
  --gateway-section-name https
```

Use `--dry-run` to validate and show the installation plan without changing the cluster. Repeatable `--values` files provide platform image, runtime, and hardware settings. Releases originally installed directly with Helm remain under their existing Helm lifecycle and are not adopted automatically.

Deploy one frontend and all models rendered by a Kustomize root:

```bash
foretoken deploy examples/multi-model-quickstart
```

The command applies the configuration, reports each `FrontendService` and `ModelService` state when it changes, and exits when every resource is Ready for its current generation. Change the default ten-minute deadline with `--timeout`.

Delete the resources rendered by the same configuration:

```bash
foretoken delete examples/multi-model-quickstart
```

The command waits for deletion and ignores resources that are already absent. After deleting all Foretoken services, remove the platform release:

```bash
foretoken uninstall
```

The command preserves Foretoken CRDs and refuses to uninstall while user-owned services remain.

Inspect the same deployment without applying it:

```bash
foretoken status examples/multi-model-quickstart
```

Inspect every Foretoken service in a namespace, or continue watching state changes:

```bash
foretoken status -n foretoken-multi-model-demo
foretoken status -n foretoken-multi-model-demo --watch
```

Resolve the public frontend URL after deployment:

```bash
FORETOKEN_FRONTEND_URL="$(foretoken endpoint examples/multi-model-quickstart)"
```

For an HTTP Gateway, resolve its request `Host` separately:

```bash
FORETOKEN_FRONTEND_URL="$(foretoken endpoint examples/quickstart)"
FORETOKEN_REQUEST_HOST="$(foretoken endpoint examples/quickstart --host)"
```

The host value is the URL authority for direct access or the configured routing hostname for an HTTP Gateway. The command waits for the LoadBalancer or Gateway address, but serving readiness remains owned by `foretoken deploy`.

Benchmark support uses optional dependencies:

```bash
pip install -e '.[bench]'
# or
uv pip install -e '.[bench]'

foretoken bench examples/quickstart
```

The CLI uses the active `kubectl` context and honors standard Kubernetes configuration such as `KUBECONFIG`.
