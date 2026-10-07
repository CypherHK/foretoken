<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
-->

# Foretoken 前端

[English](README.md) | 简体中文

前端为已部署的模型提供文本和视频生成接口。

## 文本生成

按仓库[快速开始](../../README_zh.md#快速开始)完成部署，再从仓库根目录发送请求：

```bash
DEPLOYMENT=examples/quickstart
FRONTEND_URL="$(foretoken endpoint "$DEPLOYMENT")"
REQUEST_HOST="$(foretoken endpoint "$DEPLOYMENT" --host)"

# OpenAI Responses
curl --fail-with-body "$FRONTEND_URL/v1/responses" \
  -H "Host: $REQUEST_HOST" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","input":"你好","max_output_tokens":512,"store":false}'

# Anthropic Messages
curl --fail-with-body "$FRONTEND_URL/v1/messages" \
  -H "Host: $REQUEST_HOST" \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"你好"}],"max_tokens":512}'
```

添加 `"stream": true` 和 `curl --no-buffer` 可实时接收生成内容。

| 接口 | 路径 | 请求内容 |
| --- | --- | --- |
| OpenAI Chat Completions | `POST /v1/chat/completions` | `model`、`messages` |
| OpenAI Responses | `POST /v1/responses` | `model`、`input`；设置 `store: false`，每轮携带对话历史 |
| Anthropic Messages | `POST /v1/messages` | `model`、`messages`、必填的 `max_tokens` |
| Anthropic token 计数 | `POST /v1/messages/count_tokens` | 与生成请求一致的模型、消息、系统提示和工具定义 |
| 文本补全 | `POST /v1/completions` | `model`、`prompt` |

`GET /v1/models` 列出已配置的模型标识；`/tokenize` 和 `/detokenize` 用于文本与 token ID 之间的转换。支持图片的文本模型接受 base64 编码的图片 `data:` URL。

工具由客户端执行，再将结果传入下一轮请求。Responses 支持函数工具、带命名空间的函数和自定义文本工具，不支持服务端托管工具或后台执行模式。强制选择工具和严格约束工具参数需要模型支持结构化输出。

输出 token 预算包含思考内容。Messages 使用 `max_tokens`，不接受独立的 `thinking.budget_tokens`；思考控制取决于模型的聊天模板。预算耗尽时，Messages 返回 `max_tokens`，Responses 返回 `incomplete`，客户端只应执行完整的工具调用。

## 视频生成

通过 `POST /v1/videos/sync` 提交提示词和参考图片或视频，生成完成后直接返回视频内容。[MiniMax H3 配方](../../examples/recipes/minimax-h3/a100-bf16-tp2/README_zh.md#根据图片生成视频)提供完整的部署和请求命令，将结果保存到 `./data/video.mp4`。

### 后台生成

如需提交后离开、稍后获取结果，在部署的 `frontend.yaml` 中启用：

```yaml
spec:
  videoTasks:
    claimName: video-results
    retentionSeconds: 86400
```

`video-results` 对应同一命名空间内已准备好的专用持久化存储声明（PVC）；跨节点运行时需支持 ReadWriteMany 共享访问。将参考图片放入该卷的 `inputs/reference.png`，然后应用配置：

```bash
DEPLOYMENT=examples/recipes/minimax-h3/a100-bf16-tp2
foretoken deploy "$DEPLOYMENT" --timeout 1h
FRONTEND_URL="$(foretoken endpoint "$DEPLOYMENT")"
REQUEST_HOST="$(foretoken endpoint "$DEPLOYMENT" --host)"
mkdir -p ./data
```

以下请求选择名为 `h3` 的模型服务，输入路径相对于视频存储卷根目录：

```bash
curl --fail-with-body "$FRONTEND_URL/v1/videos" \
  -H "Host: $REQUEST_HOST" \
  -H 'Content-Type: application/json' \
  -d '{
    "modelServiceRef": {"name": "h3"},
    "request": {
      "task": "fl2va",
      "prompt": "A sailboat crossing a calm bay at sunrise",
      "width": 1024, "height": 576,
      "numFrames": 124, "fps": 24, "numInferenceSteps": 50,
      "inputFiles": [{
        "field": "input_reference",
        "path": "inputs/reference.png",
        "contentType": "image/png"
      }]
    }
  }'
```

响应包含任务 `id`、状态地址 `status_url` 和视频地址 `content_url`。继续使用同一个前端地址，将下表的 `{id}` 替换为返回的任务 ID：

| 操作 | 接口 | 说明 |
| --- | --- | --- |
| 查看状态 | `GET /v1/videos/{id}` | 等待 `phase` 为 `Succeeded`；`Failed` 时查看 `reason` 和 `message` |
| 保存视频 | `GET /v1/videos/{id}/content` | 生成成功后，用 `curl --output ./data/video.mp4` 保存响应 |
| 取消任务 | `POST /v1/videos/{id}/cancel` | 请求取消，后端计算可能仍在结束中 |
| 删除任务 | `DELETE /v1/videos/{id}` | 清理任务及其文件 |

取消和删除请求返回 `202` 后继续由服务处理。以上配置保留结果一天，从任务结束起算；到期后自动清理，原始参考文件保留。

### 本地请求观测

将 `FORETOKEN_REQUEST_OBSERVATIONS_JSONL` 设置为新的本地文件路径，可开启 Router 选路观测。
父目录必须已存在；未设置时默认关闭，空路径或已存在的文件会使启动失败。Unix 文件权限为 `0600`。
不输出 prompt、token ID、响应正文、端点或凭据。使用进程内单调时钟，不同进程的偏移不能直接比较。

快照在本请求 reserve 前取得。`reserved_tokens` 保留 Router 原有未缓存输入量口径；
`pending_first_token_input_tokens` 单独统计 frontend 开始 dispatch 后、尚未收到非空 token item 的 Aggregate 文本请求完整输入量，
包括等待 backend generate 的时间；这个边界不等于引擎开始执行。
前缀比例的分母为 `floor(input_tokens / block_size)`；来源不可用或分母为零时显式记录 null 和原因。
引擎 rank gauge 保留来源采样身份，但其单调接收时间未知。

同一开关还记录 frontend 首非空 token 的 TTFT、正常 backend terminal 的 E2E、标签可见时间和事后 usage。
首 token 时写出早期 TTFT 行，结束时另写一行终止记录。失败、超时、取消、未完成流也保留，截断时长不作为正常 E2E 标签。
入口沿用协议 handler 在 JSON 解析后的时间；终止点在 backend token stream，早于 HTTP／SSE 交付，客户端口径还含解码和传输延迟。
消费者消失可能是断开、idle timeout 或 adapter 停止，实际原因记 unknown；video 不进入该标签路径。

有界写线程在路由锁外写 JSONL，队列溢出和写盘失败均计数；关闭时排空已接收行并记录计数。
丢失计数非零时不能声称数据完整。开关不改变路由评分。

## 运维

使用 `foretoken status` 查看部署状态，使用 `foretoken delete` 删除部署，均传入对应配置目录。

| 接口 | 用途 |
| --- | --- |
| `/healthz` | 进程存活状态 |
| `/readyz` | 服务就绪状态 |
| `/statusz` | 服务和缓存索引状态 |
| `/metrics` | Prometheus 指标 |

网关配置见[网关模式](../../README_zh.md#网关模式)。TLS 和身份认证由集群入口配置，运维接口的访问范围由网络策略控制。
