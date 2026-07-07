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

## 使用

```powershell
.\scripts\doctor.ps1
.\.venv\Scripts\python.exe -m zh_asr transcribe E:\path\to\audio.wav --engine sensevoice --device cuda:0 --out-dir E:\ChineseASR\outputs
.\scripts\strict.ps1 -Audio E:\path\to\audio.wav
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
- `*.strict.audit.md`：审计稿，记录两模型原文、相似度、状态、备选文本和判断依据。
- `*.strict.audit.json`：机器可读审计数据。
- `*.qwen3-asr-1.7b.raw.json` / `*.sensevoice.raw.json`：两模型原始结果。

如需临时使用另一份模型配置，可设置：

```powershell
$env:ZH_ASR_MODEL_CONFIG='E:\path\to\models.yaml'
```

## 低幻觉原则

- 静音和噪声优先交给 VAD，不让 ASR 自由生成。
- 原始逐字稿、结构化 JSON、后处理文本分层保存。
- 对低置信片段、疑似套话、模型不一致片段打标，不静默润色。
- `transcript.md` 默认保持干净；会改变语义的不确定性才内联 `[疑似]` / `[听不清]`，详细证据放进 `audit.md`。
- Whisper 只能作为交叉对照，不作为最终单一事实来源。

## 本地测试

核心测试不依赖大模型：

```powershell
$env:PYTHONPATH='E:\ChineseASR\src'
python -m unittest discover -s E:\ChineseASR\tests -v
```
