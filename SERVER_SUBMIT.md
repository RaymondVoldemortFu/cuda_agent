# 服务器提交与运行指南

## 1. 代码布局

- 入口：`/workspace/run.sh`（评测容器工作目录为 `/workspace`）。
- 实现包：`src/hw_probe/`（`python -m hw_probe.main`）。
- 依赖与锁：`pyproject.toml`、`uv.lock`；提交前在本地执行 `uv lock` 并一并上传。

## 2. 环境变量（生产 / 评测）

评测方通过**进程环境变量**注入，**不要**依赖 `/workspace/.env`（也不要把含密钥的 `.env` 提交到评测机）。

| 变量 | 说明 |
|------|------|
| `API_KEY` | 模型 API 密钥 |
| `BASE_URL` | 兼容 OpenAI 的网关地址（可为空则走默认） |
| `BASE_MODEL` | 模型名 |

可选：

| 变量 | 说明 |
|------|------|
| `HW_PROBE_WORKSPACE_ROOT` | 默认 `/workspace`（**production**；勿依赖 development 的相对路径默认） |
| `HW_PROBE_TARGET_SPEC_PATH` | 默认 `/target/target_spec.json`（**production**） |
| `HW_PROBE_SUPERVISOR_MAX_LOOPS` | 监督-编程轮次上限，默认 `12` |
| `HW_PROBE_REACT_MAX_STEPS` | 单轮 ReAct 递归深度相关，默认 `24` |

本地开发可使用：

```bash
uv run python -m hw_probe.main --dev
```

`--dev` 会在**不覆盖**已存在环境变量的前提下尝试加载仓库根目录的 `.env`。

## 3. 运行依赖（需用户或管理员准备）

以下若缺失，Agent 会在证据中记录失败原因；**不会**自动安装系统软件或修改操作系统级环境：

- `nvcc`（CUDA Toolkit）
- `ncu`（NVIDIA Nsight Compute）
- 可用 GPU 与驱动

请在具备上述条件的环境（如课程 GPU 容器或本机 4090 Linux）中验证后再提交。

## 4. 上传与提交

1. 按课程 `gpu_service_guide.md` 将工作区同步到评测机（例如 `10.176.37.31` 的 `/workspace`）。
2. 确保 `/workspace/run.sh` 可执行：`chmod +x run.sh`。
3. 提交评测（示例，替换 `<server>` 与学号）：

```bash
curl -X POST http://<server>:8080/submit \
  -H "Content-Type: application/json" \
  -d '{"id":"你的学号","gpu":1}'
```

4. 轮询：

```bash
curl http://<server>:8080/submit_status/<output_file>
```

5. 在浏览器打开 `http://<server>:8080/outputs` 下载输出。

## 5. 输入 / 输出契约

- **输入**：`/target/target_spec.json`，字段 `targets` 为字符串列表。
- **输出**：在 `HW_PROBE_WORKSPACE_ROOT`（默认 `/workspace`）下生成**唯一** `output.*` 文件，默认文件名为 `output.json`，内含：
  - `results`：各 target 对应数值或 `null`
  - `methodology`：简短流程说明
  - `evidence`：最近若干轮编程子智能体证据摘要

标准输出与错误日志见 `/workspace/results.log`（由程序追加）。

### 5.1 调试与 LLM 全链路追踪

- 目录：**`<workspace>/log/`**（由 `HW_PROBE_LOG_DIR_NAME` 控制，默认 `log`）。
- **`hw_probe.debug.log`**：`hw_probe` 命名空间下 **DEBUG** 级别应用日志（含各图节点进入/结束、配置摘要等），默认单文件最大约 20MB、保留若干备份。
- **`llm_trace.jsonl`**：一行一个 JSON 事件，记录 **chat_model_start / llm_end / tool_start / tool_end / chain_start / chain_end** 等 LangChain 回调字段（含完整 messages 序列化与 tool 输出），用于离线审计。
- 控制台默认 **INFO**；可用环境变量 `HW_PROBE_CONSOLE_LOG_LEVEL=DEBUG` 或命令行 `--console-log-level DEBUG` 提高刷屏详细度。

## 6. 智能体结构说明（便于报告撰写）

- **Planner**：仅输出实验与指标计划，不写代码。
- **编程子智能体（单 ReAct）**：合并编码与命令执行，在工作区内使用文件工具、`run_shell`、`compile_cuda`、`run_cuda_binary`、`run_ncu_profile`；提示词要求先探测环境、不假定操作系统。
- **Supervisor**：在「继续编程轮」与「汇总」之间路由。
- **Synthesizer**：从证据生成结构化 `results` JSON。

## 7. 无法运行时的排查（需用户介入）

1. `API_KEY` 未设置或无效 → 在环境或本地 `.env`（仅 `--dev`）中配置。
2. `nvcc` / `ncu` 不在 `PATH` → 由用户在运行环境安装 CUDA / Nsight Compute 或调整 `PATH`。
3. GPU 不可用 → 检查 `nvidia-smi`、容器是否申请 GPU（`"gpu":1`）。
4. 提交超时（30 分钟）→ 减少 `HW_PROBE_SUPERVISOR_MAX_LOOPS` / 缩小实验规模，或优化 prompts 减少无效工具循环。
