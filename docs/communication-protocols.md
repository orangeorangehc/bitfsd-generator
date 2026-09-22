# 前后端通信协议概览

## 1. HTTP REST

```
Client → GET /api/tracks → Server → JSON → Client
Client → POST /api/generate {preset: "oval"} → Server → 201
```

- 无状态，请求-响应模式，用完即断
- 用 HTTP method 表达语义：GET（查）、POST（增）、PUT（改）、DELETE（删）
- 数据格式通常是 JSON，浏览器 devtools 可直读
- 优点：简单、调试方便、生态成熟、curl 就能测
- 缺点：只能客户端主动拉，服务端不能主动推

适用：增删改查、表单提交、数据导出。bitfsd-generator、bitfsd-annotator 目前都用这个。

---

## 2. WebSocket

```
Client ↔ Server（持久双向连接，一次握手后长连）
Server: "训练进度 80%"
Server: "loss=0.3"
```

- `ws://` / `wss://` 协议
- 全双工，双方随时可发消息
- 需要心跳保活（ping/pong）
- 优点：真正的实时双向通信
- 缺点：比 HTTP 重，调试不如直接看 JSON，需要处理断线重连

适用：聊天室、协同标注、实时控制指令。

---

## 3. Server-Sent Events (SSE)

```
Client → GET /stream → Server（持续推送文本流）
                         data: epoch 5 loss=0.4

                         data: epoch 6 loss=0.3
```

- 单向：服务端 → 客户端，基于标准 HTTP
- 浏览器原生 `EventSource` API，不需要额外库
- 断线自动重连
- 优点：比 WebSocket 轻，实现简单（就是一个带 `text/event-stream` 头的 GET endpoint）
- 缺点：只能服务端推，客户端不能通过同一条连接发消息

适用：训练日志流、进度通知、实时指标面板。**如果想在 web 界面实时看训练曲线，这是最合适的选择。**

---

## 4. gRPC

```protobuf
// 定义在 .proto 文件中
service TrackService {
  rpc Generate(GenerateRequest) returns (Track);
  rpc Perceive(stream PointCloud) returns (stream Detection);  // 双向流
}
```

- Google 出品，基于 HTTP/2
- 数据用 Protocol Buffers 序列化（二进制），比 JSON 小且快
- `.proto` 文件自动生成多语言客户端代码（Go/Python/C++/...）
- 支持普通调用、服务端流、客户端流、双向流
- 优点：高性能、强类型契约、跨语言代码生成
- 缺点：二进制不可直读调试，浏览器不能直接调用（需要 gRPC-Web 代理）

适用：微服务间高性能通信、模型推理服务（TensorRT/ONNX 经常用 gRPC serving）。

---

## 5. WebRTC

- 点对点实时音视频 + 数据通道
- 底层用 UDP（低延迟，允许丢包）
- 涉及 STUN/TURN 服务器做 NAT 穿透

适用：视频流回传、远程操控画面、语音通话。与感知数据生成关系不大。

---

## 选型速查

| 场景 | 推荐协议 |
|------|----------|
| 普通增删改查、数据提交 | HTTP REST |
| 服务端推送（训练日志、进度） | SSE |
| 双向实时通信（聊天、协同） | WebSocket |
| 微服务间高性能通信 | gRPC |
| 点对点视频流 | WebRTC |

---

## 本项目的演进路径

当前：**HTTP REST** — 完全够用。
以后如果需要：
  - 实时显示训练曲线 → 加一个 SSE endpoint
  - 前端实时推送感知帧序列 → SSE 或 WebSocket
  - 推理服务化 → gRPC（参考 bitfsd-annotator 的 `/api/auto_label`）
