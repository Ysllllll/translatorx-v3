# TranslatorX v3

字幕 / 视频字幕的翻译平台。覆盖批量课程翻译、流式浏览器插件场景、任意语言对、术语注入、可恢复缓存和质量校验。

## 一句话上手

```bash
pip install -e .
export PYTHONPATH=src
python demos/demo_app.py            # 6 个端到端场景，需要 LLM 服务在 http://localhost:26592/v1
```

如果端口不通，demo 会打印一行说明后安静退出，不报错。

## 它能做什么

- **批量翻译一个 SRT** — 一行 builder 链
- **批量翻译一门课程下多个视频** — 自动并发，错误隔离
- **浏览器插件 / 实时流** — 喂入 segment、获取 SentenceRecord、支持 `Priority`、`seek()`、按说话人切句
- **任意语言对** — 内置 10 种语言的分词 / 分句 / 标点处理 (中 / 日 / 韩 / 英 / 俄 / 西 / 法 / 德 / 葡 / 越)
- **断点续翻** — Store + fingerprint，无变更不再耗 LLM 调用
- **术语注入** — 静态术语对 / 一次性术语 / 异步可拉取术语
- **质量校验** — 5 条规则 + 多语言 Profile，可重译

## 核心入口

```python
import asyncio
import trx

async def main():
    app = trx.App.from_dict({
        "engines": {"default": {
            "kind": "openai_compat",
            "model": "Qwen/Qwen3-32B",
            "base_url": "http://localhost:26592/v1",
            "api_key": "EMPTY",
        }},
        "contexts": {"en_zh": {"src": "en", "tgt": "zh", "window_size": 4}},
        "store": {"kind": "json", "root": "./workspace"},
    })

    # 单视频
    result = await (
        app.video(course="cs101", video="lec01")
        .source("lec01.srt", language="en")
        .translate(src="en", tgt="zh")
        .run()
    )
    for rec in result.records:
        print(rec.src_text, "->", rec.translations["zh"])

asyncio.run(main())
```

更多模式请看 [`demos/demo_app.py`](demos/demo_app.py) 里的 6 个场景：
1. 单 SRT (VideoBuilder)
2. 课程批量 (CourseBuilder)
3. 实时流 + 优先级 + seek (StreamBuilder)
4. 断点续翻 (fingerprint cache 命中)
5. 自定义错误上报 (ErrorReporter)
6. 多说话人流 (split_by_speaker)

## 架构总览

8 层包，依赖严格自上而下：

```
L0  model/          数据类型 (Word / Segment / SentenceRecord)
L1  lang_ops/       多语言文本操作
L1  media/          音视频下载 + ffmpeg
L2  subtitle/       字幕解析 / 词时间对齐 / 句重组
L2  llm_ops/        LLM 引擎 + 翻译微循环 + 术语
L2  checker/        翻译质量校验
L3  runtime/        Processor / Store / Orchestrator (核心引擎)
L3  trx/            统一门面 (Facade)
```

完整说明: [`CLAUDE.md`](CLAUDE.md) (架构 + 命名约定 + 测试位置)，[`src/runtime/README.md`](src/runtime/README.md) (核心引擎细节)。

## 安装

依赖 Python 3.10+ (使用 `slots=True` / `frozen=True` / `str | None` 等语法)。

```bash
pip install -e .
```

可选依赖（按需）：
- `jieba` — 中文分词（CJK 测试需要）
- `mecab-python3` + `unidic-lite` — 日语分词（缺失时相关测试 skip）
- `kiwipiepy` — 韩语分词

LLM 服务（自由选择 OpenAI 兼容端点）：

```python
{"kind": "openai_compat", "model": "...", "base_url": "...", "api_key": "..."}
```

## 测试

```bash
pytest tests/ -q                                # ~8s, 813 通过 / 2 skip
pytest tests/runtime_tests/ -v                  # 只跑 runtime 层
pytest tests/ --cov=src --cov-report=term       # 覆盖率
```

## 演示

| 文件 | 说明 |
|---|---|
| `demos/demo_lang_ops.py` | 分词 / 分句 / ChunkPipeline |
| `demos/demo_subtitle.py` | SRT 解析 / Word 对齐 / Subtitle 重组 |
| `demos/demo_checker.py`  | 翻译质量检查 |
| `demos/demo_media.py`    | yt-dlp 下载 + ffmpeg 提取 |
| `demos/demo_llm_ops.py`  | LLM 引擎 + 上下文 + translate_with_verify (需要 LLM) |
| `demos/demo_app.py`      | 端到端 6 场景 (需要 LLM) |

## 当前进度

- ✅ 翻译子系统（批量 + 流式）端到端可用
- ⬜ 转录 (Stage 6 — Transcriber/Align Processor)
- ⬜ 配音 / TTS (Stage 6 — TTSProcessor + VoicePicker)
- ⬜ HTTP 服务 / 多用户 / 鉴权 / 限流 (Stage 7 — FastAPI + SSE)

设计决策完整记录见 `processor-architecture-memo.md` (D-001..D-068)。
