# AGENTS.md

面向在本仓库工作的 AI 编程智能体的指引。

## 语言约定

本仓库所有面向人的文档（AGENTS.md、CLAUDE.md、GLOSSARY.md、docs/ 下的说明）、提交信息、以及代码注释中的说明性文字，统一使用中文撰写。仅保留无法翻译的标识符原文：GitHub 标签名（如 `needs-triage`）、命令（如 `gh`）、文件路径、API / 符号名。后续新增文档默认沿用中文。

## Agent skills

### Issue tracker（问题追踪器）

本仓库的 issue 与 spec 都以 GitHub issues 形式存在，相关技能使用 `gh` CLI 进行操作。详见 `docs/agents/issue-tracker.md`。

### Triage 标签

采用五个规范的 triage 角色，标签字符串与其名称一致（`needs-triage`、`needs-info`、`ready-for-agent`、`ready-for-human`、`wontfix`）。详见 `docs/agents/triage-labels.md`。

### Domain docs（域文档）

单上下文布局：仓库根目录一个 `GLOSSARY.md`，外加 `docs/adr/`。详见 `docs/agents/domain.md`。

## 构建环境

本仓库依赖 VS（MSVC）、OpenCV、Eigen、Python；各机器路径不同，故**不写死绝对路径**：VS/cmake/ninja 由 `vswhere` 自动定位，依赖位置由各机器通过环境变量或 `scripts/configure.local.bat`（git 忽略）注入。可移植构建命令、关键坑（OpenCV_RUNTIME=vc16、USE_CUDA 默认 OFF、SBM_SIMD_LEVEL 跨机器注意）、本地偏离项见 `docs/agents/build.md`。
