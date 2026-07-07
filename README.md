# ChineseASR

本项目是 9950X3D + RTX 5090 D 本地中文语音转文字项目，目标是 **中文低幻觉、可审计、可复测**。

## 结论

- strict 最优双模型路线：`Qwen3-ASR-1.7B + SenseVoiceSmall`
- quick / 低成本路线：`SenseVoiceSmall + FSMN-VAD + ct-punc + cam++`
- 保守备用路线：`Paraformer-zh`
- Whisper：只做 fallback / comparison，不作为中文严谨转写主模型

原因：公开评测和官方模型卡更支持 Qwen3-ASR-1.7B 作为准确率优先模型；SenseVoiceSmall 继续作为高速、低幻觉的声学锚点；Paraformer 保留用于普通话生产基线、时间戳、热词和回归对照。Whisper 在静音、噪声、长停顿里有整句幻觉风险，不作为中文严谨转写主轴。

## 网络与下载约束

所有脚本都会先清空这些代理变量：

```powershell
HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, http_proxy, https_proxy, all_proxy
```

脚本会保留本地和阿里云 / ModelScope 的 `NO_PROXY`，模型缓存固定在：

```text
E:\ChineseASR\models\modelscope
```

模型预热后，运行时会优先把 ModelScope 模型 ID 解析为本地缓存路径，尽量避免每次转写都联网检查文件列表。

如果你要完全避免外网下载，请先把 PyTorch / FunASR / ModelScope wheel 放到本地 wheelhouse，再用 `scripts\install-torch-cu128-direct.ps1 -Wheelhouse <目录>`。

项目也提供轻量离线恢复链路，用于把当前可工作的 Python 依赖冻结成 lock，再下载 wheel 并生成 SHA256 校验：

```powershell
.\scripts\export-lock.ps1
.\scripts\build-wheelhouse.ps1
.\scripts\verify-wheelhouse.ps1
.\scripts\install-offline.ps1 -Venv E:\ChineseASR\.venv-offline-smoke
```

`build-wheelhouse.ps1` 会先清空代理环境变量，PyTorch 走 `https://download.pytorch.org/whl/cu128`，普通 Python 包默认走清华 PyPI 源。`offline\wheelhouse\` 只存大 wheel 文件并被 Git 忽略；`offline\manifests\` 用于保存 `requirements-lock.txt`、`wheelhouse.sha256` 和 `wheelhouse.json` 这类小清单。第一版只覆盖 Python wheel 依赖，模型权重仍由 ModelScope 缓存和 `download-models.ps1` 管理。

## 安装顺序

在 PowerShell 中运行：

```powershell
cd E:\ChineseASR
.\scripts\install-torch-cu128-direct.ps1
.\scripts\setup-core.ps1
.\scripts\download-models.ps1 -Engine sensevoice
```

`install-torch-cu128-direct.ps1` 默认使用 PyTorch 官方 CUDA 12.8 wheel 源，并显式禁用代理。RTX 5090 D / Blackwell 通常需要 CUDA 12.8 或更新的 PyTorch 构建；如果稳定源不适配，可加 `-Nightly` 切换 PyTorch nightly cu128。

strict 默认使用 `qwen3-asr-1.7b + sensevoice`。首次启用 Qwen 路线时再运行：

```powershell
.\scripts\setup-qwen.ps1
.\scripts\download-models.ps1 -Engine qwen3-asr-1.7b
```

`download-models.ps1 -Engine qwen3-asr-1.7b` 会先通过 ModelScope 下载 `Qwen/Qwen3-ASR-1.7B` 到 `E:\ChineseASR\models\modelscope\Qwen\Qwen3-ASR-1.7B`，再 warmup，避免运行时默认走 Hugging Face 自动下载。

Qwen adapter 会强制使用这个本地目录；如果目录不存在，会直接报错并提示先预取，不会把模型 ID 交给 runtime 自行联网。
Qwen 会收到“只输出简体中文”的上下文提示；同时输出层会用 OpenCC 统一转为简体中文，原始模型文本保存在 raw JSON 的 `original_text` 字段。

## 使用

```powershell
.\scripts\doctor.ps1
.\.venv\Scripts\python.exe -m zh_asr transcribe E:\path\to\audio.wav --engine sensevoice --device cuda:0 --out-dir E:\ChineseASR\outputs
.\scripts\strict.ps1 -Audio E:\path\to\audio.wav
.\scripts\asr-smart.ps1 -Audio E:\path\to\audio.wav -WaitSec 15
.\scripts\transcribe-folder.ps1 -InputDir E:\path\to\audio-folder
.\scripts\eval.ps1 -Generate
.\scripts\benchmark.ps1 -AudioDir E:\path\to\audio -TruthDir E:\path\to\truth
```

模型注册表在：

```text
E:\ChineseASR\configs\models.yaml
```

默认 quick 模型、strict 双模型、FunASR 模型 ID 和 VAD / 标点 / 说话人模型别名都从这个文件读取。同一 `funasr` 适配器内替换或新增模型时，优先只改 YAML；跨运行时模型才需要新增 adapter。

`transcribe` 是 quick 模式，输出包含：

- `*.sensevoice.md`：可读 Markdown 转写
- `*.sensevoice.raw.json`：原始 JSON，保留时间戳和说话人字段

`strict` 是重要音频模式，默认会顺序运行 `qwen3-asr-1.7b` 和 `sensevoice`，输出：

- `*.strict.md`：最终稿。正文尽量干净，只有严重分歧或听不清时才内联 `[疑似]` / `[听不清]`。
- `*.strict.audit.md`：审计稿，记录两模型原文、相似度、状态、备选文本、`rule_hits` 和判断依据。
- `*.strict.audit.json`：机器可读审计数据，包含 `flags` 和结构化 `rule_hits`。
- `*.qwen3-asr-1.7b.raw.json` / `*.sensevoice.raw.json`：两模型原始结果。

`transcribe-folder.ps1` 是日常批量入口，默认递归扫描 `wav/mp3/m4a/flac`，按每个音频单独建输出目录，并运行 strict 双模型。已经存在 `*.strict.md` 的文件会自动跳过；加 `-Force` 可重跑。批量输出包括：

- `summary.md`：总数、已处理、已跳过、失败数和每个文件的输出目录。
- `failed.jsonl`：失败文件、错误类型和错误信息，便于修复后重跑。
- 每个音频子目录中的 strict / raw JSON / audit 文件。

如果只想快速粗转，可用：

```powershell
.\scripts\transcribe-folder.ps1 -InputDir E:\path\to\audio-folder -Mode quick -Engine sensevoice
```

`eval.ps1` 是隐私友好的评测入口，不要求你提供真实录音。它会生成本地可复现的评测语料：

- `synthetic`：Windows 本地中文 TTS 生成的已知答案语音。
- `adversarial`：静音、白噪声、纯音等负样本，用来测幻觉底线。
- `truth`：标准答案文本；负样本标准答案为空。

只生成语料、不跑模型：

```powershell
.\scripts\eval.ps1 -Generate -GenerateOnly
```

完全跳过 TTS，只生成负样本：

```powershell
.\scripts\eval.ps1 -Generate -GenerateOnly -NoTts
```

完整评测会复用 strict 双模型，输出：

- `metrics.json`：schema v2 运行账本，含模型配置快照、命令、运行环境、耗时、CER、模型分歧、文本相似度、风险标记、`rule_hits`、false confident 统计。
- `benchmark.md`：整体分数表。
- `review.md`：按 `P0/P1/P2` 排序的人工复核队列，会列出原因、建议动作、音频/audit/raw JSON 路径和截断文本证据。

`benchmark.ps1` 用于已有人工标准答案的真实/公开/第三方音频批次。音频和标准答案按文件名 stem 匹配：

```text
audio\
  001.wav
  002.mp3
truth\
  001.txt
  002.txt
```

运行后会调用 strict 双模型，并输出：

- `benchmark.json`：机器可读结果，复用 schema v2 运行账本，并补充 benchmark 的音频目录、truth 目录和 manifest 路径。
- `benchmark.md`：人工可读汇总表。
- `review.md`：缺 truth、模型分歧、疑似幻觉和 false confident 样本；`P0` 优先回听，`P1` 核对后再信任，`P2` 补 truth 或确认跳过。

`benchmark.ps1` 不会复制你的源音频或标准答案，只在输出目录下写 `_manifest\manifest.json` 记录路径、音频/truth hash、模型配置快照和运行清单。缺少对应 `.txt` 的音频不会评分，会进 `review.md`。

## 本地 API 和 smart 入口

给 Codex / 自动化调用时，优先使用 smart 入口，避免长时间卡住命令行：

```powershell
.\scripts\asr-smart.ps1 -Audio E:\path\to\audio.wav -Mode strict -WaitSec 15 -Json
```

`asr-smart.ps1` 会自动启动本地 API：

```powershell
.\.venv\Scripts\python.exe -m zh_asr serve --host 127.0.0.1 --port 8765
```

API 只绑定 `127.0.0.1`，支持：

- `GET /health`：服务状态、队列长度、GPU 冲突。
- `GET /jobs`：任务列表。
- `GET /jobs/{job_id}`：任务状态和输出路径。
- `POST /jobs/transcribe`：提交 quick / strict 转写。
- `POST /jobs/{job_id}/cancel`：取消排队或运行中的任务。

服务内部只有一个 GPU worker，同一音频和同一配置会自动去重。任务在子 Python 进程里跑现有 CLI，所以 API 请求不会被模型推理长期占住，取消任务时也可以终止子进程。

默认会用 `nvidia-smi` 检测外部 CUDA compute 进程；如果发现 Ollama、LocalOCR、LM Studio、其他 Python 模型进程等正在占 GPU，新任务会返回 `blocked`，不会硬抢显存。你确实想覆盖时再显式加：

```powershell
.\scripts\asr-smart.ps1 -Audio E:\path\to\audio.wav -AllowGpuConflicts
```

如需临时使用另一份模型配置，可设置：

```powershell
$env:ZH_ASR_MODEL_CONFIG='E:\path\to\models.yaml'
```

## 低幻觉原则

- 静音和噪声优先交给 VAD，不让 ASR 自由生成。
- 原始逐字稿、结构化 JSON、后处理文本分层保存。
- 对低置信片段、疑似套话、模型不一致片段打标，不静默润色。
- deterministic 风险规则会标记：`empty_audio_hallucination`、`suspicious_stock_phrase`、`abnormal_repetition`、`model_conflict`、`traditional_residue`、`long_unpunctuated_text`。
- `model_conflict` 只在两路模型都有实质文本且相似度低时触发；单边空输出优先交给静音/模板幻觉规则判断。
- `transcript.md` 默认保持干净；会改变语义的不确定性才内联 `[疑似]` / `[听不清]`，详细证据放进 `audit.md`。
- Whisper 只能作为交叉对照，不作为最终单一事实来源。

## 本地测试

核心测试不依赖大模型：

```powershell
$env:PYTHONPATH='E:\ChineseASR\src'
python -m unittest discover -s E:\ChineseASR\tests -v
```
