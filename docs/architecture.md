# 架构说明

## 目标

`ChineseASR` 面向中文严谨转写：先保证不幻觉、不乱编，再追求速度和格式化效果。

## 引擎策略

1. `sensevoice`：默认主线。组合 `iic/SenseVoiceSmall`、`fsmn-vad`、`ct-punc`、`cam++`。
2. `paraformer`：中文保守对照线。适合做交叉识别和基准回归。
3. `whisper-large-v3`：只记录为 fallback/comparison，不自动作为主输出。

## 数据流

```text
audio -> VAD split -> ASR engine -> raw JSON -> Markdown transcript
```

后续扩展时应增加：

- 音频 hash 和输入元数据。
- 双模型交叉验证。
- 疑似幻觉片段标记。
- CER/WER/CER-like 中文评测集。

## 下载策略

脚本层和 Python 层都会清空代理变量。模型默认走 ModelScope ID，并缓存到 `E:\ChineseASR\models\modelscope`。

