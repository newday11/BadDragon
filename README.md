# BadDragon

Minimal agent scaffold with clear layered architecture:

- `app/interfaces`: CLI/API/UI entrypoints
- `app/orchestrator`: loop and state machine
- `app/llm`: model gateway
- `app/tools`: tool definitions and dispatcher
- `app/memory`: memory read/write and injection
- `app/infra`: config/logging/storage helpers
- `data`: local runtime data (e.g. sqlite)
- `logs`: runtime logs
- `tests`: tests

## Six-Layer Architecture (Required)

BadDragon development must follow this fixed six-layer design:

1. `interfaces`
   Only handles input/output boundaries (CLI/API/UI). No business logic.
2. `orchestrator`
   Only handles task orchestration, turn loop, and state transitions.
3. `llm`
   Only handles model requests, responses, and tool-call parsing.
4. `tools`
   Only implements executable tools and tool registry/dispatching.
5. `memory`
   Only handles memory read/write and prompt injection strategy.
6. `infra`
   Only handles shared infrastructure (config, logging, storage, utilities).

### Boundary Rule

Use strict one-way dependencies:

`interfaces -> orchestrator -> (llm/tools/memory) -> infra`

No reverse coupling across layers.

## 六大层级规范（必须遵守）

后续开发统一按以下六层实施，每层一个目录，禁止职责混写：

1. `interfaces`：只负责输入输出（CLI/API/UI）。
2. `orchestrator`：只负责编排、循环和状态机。
3. `llm`：只负责模型通信与返回解析。
4. `tools`：只负责工具实现与注册分发。
5. `memory`：只负责记忆读写与注入策略。
6. `infra`：只负责配置、日志、存储和通用基础能力。

依赖方向固定为：

`interfaces -> orchestrator -> (llm/tools/memory) -> infra`

## Run

Run directly in project directory:

```bash
python3 -m app.main
```

Quick payload capture test:

```bash
python3 -m app.main --hello
```

## Dependency Strategy

Follow GenericAgent-style dependency management:

1. Keep minimal required dependencies only.
2. Start first, then install optional packages when needed.
3. Do not preinstall all possible packages.

Notes:

1. `openai` SDK is optional in current version (HTTP fallback exists).
2. Future tools/frontends should be added as optional, not forced by default.

## Model Config

1. Prefer editing `config.json` (zero third-party dependency).
2. Set `model.default`, `model.provider`, `model.base_url`, `model.api_key`.
3. Runtime provider is resolved by `app/llm/runtime_provider.py`.
4. Current default uses OpenAI-compatible `chat.completions`.
