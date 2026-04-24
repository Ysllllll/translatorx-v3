# demo_stream_preprocess — 流式预处理演示

WebSocket 服务端 + 模拟客户端，用来验证**浏览器插件场景**中
“边抓字幕、边送到后端做标点恢复 + 分句 + 细切”的前半段流水线。

## 架构

```
客户端 (浏览器插件 / Python client)                                ↓   WebSocket
                         {type:"segment", start, end, text}                                ↓
   ┌─────────── server.py ─────────────────────┐
   │ PuncBufferStage  (可选，累积N条做标点恢复)  │
   │          ↓                                 │
   │ PushQueueSource (Subtitle.stream 切完整句) │
   │          ↓                                 │
   │ PreprocessProcessor  (.clauses + chunk)    │
   └────────── ↓  {type:"record", ...} ─────────┘
                         ↑   WebSocket
客户端（打印或再走翻译）
```

- **`PuncBufferStage`**：原始 ASR 没有标点，会让 `Subtitle.stream()`
  永远不切句。所以服务端先缓冲 `window` 条 cue，合并后丢给
  `PuncRestorer`，恢复好标点再送入 `PushQueueSource`。
- **`PushQueueSource`**（`src/adapters/sources/push.py`）：以 asyncio
  队列为 backing store，每完成一整句就 yield 一个 `SentenceRecord`。
- **`PreprocessProcessor`**：对每条完成的 `SentenceRecord` 再跑
  `clauses()` + `transform(chunk_fn, scope="chunk")`，把长 clause
  按 `max_len` 细切。符合现有 `Processor` 契约的精简版。

## 消息协议

客户端 → 服务端（JSON）：

| `type`    | 字段                              | 说明                       |
|-----------|----------------------------------|----------------------------|
| `segment` | `start, end, text, speaker?`     | 推送一条字幕               |
| `flush`   | —                                | 立即 drain 缓冲窗口        |
| `close`   | —                                | 结束流，服务端将 flush+done |

服务端 → 客户端：

| `type`   | 字段                                                         |
|----------|-------------------------------------------------------------|
| `ready`  | `language, restore_punc, window, max_len`                   |
| `record` | `id, start, end, src_text, segments[{start,end,text,words}]` |
| `error`  | `message`                                                   |
| `done`   | —                                                           |

查询串（连接时）：`language=en&restore_punc=true&max_len=60&window=4`

## 运行

### 1. mock 模式（无外部依赖）

终端 A：
```bash
python demos/demo_stream_preprocess/server.py
```

终端 B：
```bash
python demos/demo_stream_preprocess/client.py
# 或用真 SRT + 实时节奏
python demos/demo_stream_preprocess/client.py --srt /path/to/foo.srt --paced
```

### 2. 真 backend

```bash
python demos/demo_stream_preprocess/server.py \
        --real \
        --engine http://localhost:26592/v1
```

`--real` 会：
- 用 `deepmultilingualpunctuation` 做标点恢复（需 `pip install`）。
- 用 `spacy` + LLM + rule 三段式 composite chunker。

### 3. 从浏览器 JS 调用

```js
const ws = new WebSocket("ws://127.0.0.1:8765/ws/preprocess?language=en");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.onopen = () => {
    ws.send(JSON.stringify({type:"segment", start:1.0, end:4.5,
                            text:"hello everyone welcome to the course"}));
    // ... 更多 segment
    ws.send(JSON.stringify({type:"close"}));
};
```

## 注意事项

- 当前 demo **不持久化**（`Processor` 契约中的 `store/video_key/ctx`
  省略）。要接入真实 pipeline，请用 `StreamingOrchestrator`
  （`src/application/orchestrator/video.py:243`）。
- `PuncBufferStage` 合并后丢失了窗口内单 cue 的文本→时间微对齐；
  词级 `words` 透传未动。若 ASR 已带单词时间戳，
  `SentenceRecord.segments[*].words` 会正常聚到对应 chunk 上。
- 想关闭标点恢复（SRT 已带标点）：客户端加 `--no-restore-punc`。
