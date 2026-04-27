# TranslatorX v3 — Refactor 文档

Pipeline-DSL 重构（C → C+D → C+D+F → 流式 SaaS）的设计、路线、历史快照。

## 当前进度

- **HEAD**：`f25b012`
- **测试套**：2226 passed / 3 skipped
- **最近交付**：Phase 4 — Redis Streams 跨进程 bus（J） + WebSocket 双向协议（K）
- **下一切片**：Phase 5 — Tenant Scheduler + 分级架构（方案 L + M）

详见 [`ROADMAP.md`](ROADMAP.md)。

## 文档结构

```
docs/refactor/
├── README.md                       # 本文件 — 索引 + 当前快照
├── ROADMAP.md                      # 路线状态：已完成 / 进行中 / 计划中
├── design/                         # ★ 长期参考的设计文档
│   ├── options.md                  # 主方案对比（A–H 八方案 + C/D/F 路线推荐）
│   └── streaming.md                # 流式深化（方案 I/J/K/L/M + 7 阶段路线）
└── history/                        # 已完成阶段的设计快照（不再更新）
    ├── kickoff.md                  # 启动 brief（Phase 1 启动时使用）
    ├── phase1-architecture.md      # Phase 1 (C) 架构详细设计
    └── phase1-deep-dive.md         # Phase 1 (C) 实施 deep dive
```

## 阅读路径

- **第一次接触本仓库**：`ROADMAP.md` → `design/options.md` §0–2 → `design/streaming.md` §1–3
- **要做下一步规划**：`ROADMAP.md` → `design/streaming.md` §8（剩余阶段路线）
- **想理解当前 PipelineRuntime 设计**：`history/phase1-architecture.md` + `history/phase1-deep-dive.md`
- **要写新 stage / 新 backend**：`docs/plugin_sdk.md`（不在本目录）

## 维护约定

- `ROADMAP.md` — 每个里程碑收尾时同步更新（"已完成"加条目，bump 测试套基线）。
- `design/*.md` — 设计变更或新阶段细化时编辑；保持"长期可读"风格。
- `history/*.md` — 冻结，不再修改。仅作为决策审计 / 上下文回溯。
