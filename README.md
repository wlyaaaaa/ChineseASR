# ChineseASR

本项目是 9950X3D + RTX 5090 D 本地中文语音转文字项目，目标是 **中文低幻觉、可审计、可复测**。

## 结论

- 主模型路线：`SenseVoiceSmall + FSMN-VAD + ct-punc + cam++`
- 对照路线：`Paraformer-zh`
- Whisper：只做 fallback / comparison，不作为中文严谨转写主模型

原因：Whisper 在静音、噪声、长停顿里有整句幻觉风险；本项目优先使用 FunASR 生态里更适合中文和 VAD 分段的模型。

## 网络与下载约束

所有脚本都会先清空这些代理变量：

```powershell
HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, http_proxy, https_proxy, all_proxy
```

脚本会保留本地和阿里云 / ModelScope 的 `NO_PROXY`，模型缓存固定在：

```text
E:\ChineseASR\models\modelscope
```

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

## 使用

```powershell
.\scripts\doctor.ps1
.\.venv\Scripts\python.exe -m zh_asr transcribe E:\path\to\audio.wav --engine sensevoice --device cuda:0 --out-dir E:\ChineseASR\outputs
```

输出包含：

- `*.sensevoice.md`：可读 Markdown 转写
- `*.sensevoice.raw.json`：原始 JSON，保留时间戳和说话人字段

## 低幻觉原则

- 静音和噪声优先交给 VAD，不让 ASR 自由生成。
- 原始逐字稿、结构化 JSON、后处理文本分层保存。
- 对低置信片段、疑似套话、模型不一致片段打标，不静默润色。
- Whisper 只能作为交叉对照，不作为最终单一事实来源。

## 本地测试

核心测试不依赖大模型：

```powershell
$env:PYTHONPATH='E:\ChineseASR\src'
python -m unittest discover -s E:\ChineseASR\tests -v
```

