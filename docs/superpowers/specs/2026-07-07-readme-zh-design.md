# README 中文化设计

## 目标

把公开仓库入口 `README.md` 改成简体中文主文档，让第一次打开仓库的人能直接理解 ChineseASR 的定位、默认模型、安装、常用命令、输出文件、API、长音频、评测、离线依赖、模型替换和公开边界。

## 范围

- 修改 `README.md` 为中文完整版。
- 保留命令、路径占位符、模型名、API 端点、配置键、文件名和字段名的英文原文。
- 不修改 Python 代码、PowerShell 脚本、模型配置或现有架构文档。
- 不写入本机私人音频、真实输出内容、绝对私有路径、token、日志或模型权重信息。

## 结构

README 采用“先能用，再解释”的顺序：

1. 项目定位和当前状态。
2. 适合/不适合场景。
3. 默认模型策略。
4. 一分钟使用和安装下载。
5. 输出文件、本地 API、长音频、LLM 仲裁。
6. 评测、离线 wheelhouse、模型替换、测试。
7. 公开仓库边界、常见问题、更多文档。

## 验证

改动后运行：

- `.\.venv\Scripts\python.exe -m unittest discover -s tests`
- `.\scripts\doctor.ps1`
- `git diff --check`
- README / docs / src / tests / scripts / configs 的公开秘密模式扫描

不重复跑重模型 smoke，除非 README 命令、模式名或脚本入口发生不确定变更。本次只改文档，不改变运行行为。
