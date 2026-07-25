# 架构说明

## 目标

`ChineseASR` 面向中文严谨转写：先保证不幻觉、不乱编，再追求速度和格式化效果。

## 引擎策略

引擎不再硬编码在 Python 实现里，而是注册在 `configs/models.yaml`：

1. `defaults.engine` 决定 quick 模式默认模型。
2. `strict.primary_engine` / `strict.secondary_engine` 决定严格模式双模型组合。
3. `aliases` 记录 VAD、标点、说话人等可复用模型别名。
4. `engines.*.adapter` 决定运行时适配器；当前已实现 `funasr` 和 `qwen-asr`。

当前默认策略仍然是：

1. `qwen3-asr-1.7b`：strict 准确率优先主线。基于 Qwen3-ASR 官方开源权重和 `qwen-asr` runtime。
2. `sensevoice`：quick 默认和 strict 低幻觉锚点。组合 `iic/SenseVoiceSmall`、`fsmn-vad`、`ct-punc`、`cam++`。
3. `paraformer`：中文保守备用线。适合普通话生产基线、时间戳、热词和回归对照。
4. `whisper-large-v3`：只记录为 fallback/comparison，不自动作为主输出。

## 数据流

```text
audio -> VAD split -> ASR engine -> raw JSON -> Markdown transcript
```

严格模式：

```text
audio
  -> Qwen3-ASR raw JSON
  -> SenseVoice raw JSON
  -> normalized text comparison
  -> strict.md + strict.audit.md + strict.audit.json
```

`strict.md` 是给人读的最终稿，正文尽量干净；当两个模型严重冲突、都为空、或出现常见幻觉套话时才写入 `[疑似]` / `[听不清]`。`strict.audit.md` 保存两模型原文、相似度、备选文本和判断依据。后续接入顶级 LLM 仲裁时，应只读取这个审计输入，不覆盖原始 ASR 证据。

如果 strict 中某个引擎抛错，流程不会直接丢弃整次任务。成功的一路会继续进入最终猜测，正文标记 `[疑似]`，审计报告状态为 `engine_failure`，并在 raw JSON 中保留失败引擎、异常类型和错误摘要。如果两路都失败，则输出 `[听不清]` 并要求人工复核。

长音频模式：

```text
long audio
  -> deterministic chunks + manifest.json
  -> each chunk runs strict
  -> skip completed chunks on resume
  -> optional uncertain-only Ollama arbitration
  -> transcript.md + audit.md + metrics.json
```

第一版长音频切片采用固定时长和 overlap，默认 `chunk_sec=300`、`overlap_sec=1`。`manifest.json` 记录音频 hash、模型配置 hash、chunk 参数和每个 chunk 的状态；同一输入重跑时，已成功且输出存在的 chunk 会跳过，残留 `running` 会视为 stale 并重跑。后续可以把 planner 替换成 VAD 静音边界切片，但 manifest/schema 不变。

LLM 仲裁默认关闭，配置在 `configs/models.yaml` 的 `llm_arbitration`。当前 provider 是本地 Ollama，默认模型 `qwen-main-v1:latest`，`keep_alive=0`。仲裁只读取 chunk audit 证据，不读取音频；只在 `flags`、`needs_review` 或低相似度 chunk 上触发。仲裁结果写入 merged audit / metrics，不覆盖原始 strict raw JSON。

## 本地 API / Smart 调用层

面向 Codex 和自动化调用时，不直接长时间等待 `strict.ps1`。推荐路径是：

```text
scripts\asr-smart.ps1
  -> 127.0.0.1 local API
  -> in-memory job queue
  -> single GPU worker
  -> child process: python -m zh_asr strict/transcribe
  -> outputs\api\<job_id>\
```

API 入口由 `python -m zh_asr serve --host 127.0.0.1 --port 18666` 提供，主要端点是 `/health`、`/jobs`、`/jobs/{job_id}`、`/jobs/transcribe` 和 `/jobs/{job_id}/cancel`。`/observer/jobs` 与 `/observer/jobs/{job_id}` 提供稳定的只读安全投影，只暴露调度元数据、可验证的 chunk 计数和终态 RTF，不暴露输入/输出路径、进程与命令信息、日志、转写正文或 Broker 细节。

服务层只负责调度，不在 HTTP 请求线程中加载模型。每个任务在独立 Python 子进程里运行现有 CLI，这样 API 可以快速返回、任务可以取消、模型显存也不会长期留在 API 进程里。

为避免和其他本地模型互相抢 GPU，提交任务前会尝试读取 `nvidia-smi --query-compute-apps`。发现外部 CUDA compute 进程时默认返回 `blocked`；只有显式传入 `allow_gpu_conflicts=true` 或 `scripts\asr-smart.ps1 -AllowGpuConflicts` 才会继续入队。服务不会自动终止 Ollama、LocalOCR、LM Studio 或其他 Python 模型进程。

当前机器是 RTX 5090D 32GB，默认排他锁属于保守调度策略，不是硬件能力判断。需要并发本地模型时，用显式 override，而不是改默认安全边界。

固定端到端验收入口是：

```powershell
.\scripts\smoke-asr-smart.ps1 -Json
```

它会使用本地模型缓存自带的中文样例音频，强制跑一次 strict smart job，并校验 `final`、`audit`、`audit_json`、`primary_raw_json` 和 `secondary_raw_json` 都存在。

## 模型替换边界

同一 adapter 内替换模型时，优先只改 `configs/models.yaml`，然后运行：

```powershell
.\scripts\download-models.ps1 -Engine <engine-name>
.\scripts\strict.ps1 -Audio C:\path\to\audio.wav
```

新增不同运行时，例如 Whisper 本地实现、其他 LLM 音频模型或云 API 时，应新增 adapter，并保持 pipeline 只依赖统一的 `build_model(...)` / `generate(...)` 形状。当前已有 `funasr` 和 `qwen-asr` 两个 adapter。

## 已实现的审计闭环

当前个人使用版已经闭合以下能力，作为已交付范围记录：

- 输入与运行可追踪：`manifest.json` / `metrics.json` 记录音频 hash、truth hash、模型配置 hash、选中模型、命令、运行时和耗时。
- 双模型审计：strict 输出 `strict.md`、`strict.audit.md`、`strict.audit.json` 和两路 raw JSON，最终稿与证据分离。
- 幻觉规则库：静音出字、常见模板废话、异常重复、双模型大分歧、繁体残留、超长无标点都会进入 audit / metrics / review。
- 评测与 benchmark：内置隐私友好的合成/对抗评测集；用户私有音频可通过同名 audio/truth 目录跑 `benchmark.md`、`benchmark.json`、`review.md`。
- 人工复核队列：`review.md` 按 P0/P1/P2 排序，给出复核原因、建议动作、截断证据和源文件路径。
- 可选 LLM 仲裁：长音频模式可接本地 Ollama，只对不确定 chunk 做 evidence-only 仲裁，结果写入 audit / metrics，不改写 raw ASR 证据。

LLM 仲裁刻意默认关闭。这是资源和可信度边界：默认转写链路必须在没有 Ollama、没有额外 GPU 驻留、没有顶级模型猜测的情况下稳定工作；需要最终猜测时再显式打开。

固定时长长音频切片也不是当前关闭阻塞项。现有 manifest/schema 已支持断点续跑和未来替换 planner；后续如果真实长录音暴露边界切断问题，可以把 planner 升级为 VAD 静音边界切片，而不改变输出契约。

个人使用版关闭标准：

1. `doctor.ps1` 能确认无代理、CUDA、模型配置和依赖状态。
2. 单元测试全通过。
3. `smoke-asr-smart.ps1 -Json` 能完成 strict smart job，并产出 final、audit、audit JSON 和两路 raw JSON。
4. 公开仓库只包含源码、脚本、配置、测试和文档，不包含模型权重、用户音频、输出转写或 wheelhouse 大文件。

真实私人录音 benchmark 属于后续校准，不是公开项目或本地工具链的关闭阻塞项。

## 下载策略

脚本层和 Python 层都会清空代理变量。模型默认走 ModelScope ID，并缓存到项目根目录下的 `models\modelscope`。

运行时会优先检查缓存目录：如果 YAML 中的模型 ID 已在本地缓存中，就把它们作为本地路径传给 FunASR，减少日常转写时的网络探测。

Qwen3-ASR 权重必须先用 `scripts\download-models.ps1 -Engine qwen3-asr-1.7b` 预取。该脚本使用 ModelScope 的 `Qwen/Qwen3-ASR-1.7B`，本地目录是 `models\modelscope\Qwen\Qwen3-ASR-1.7B`。
