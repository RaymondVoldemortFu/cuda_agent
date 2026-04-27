以下是润色后的实验报告，已去除所有课程相关描述（如“根据课程文档”、“本学期的核心方向”等），仅保留技术架构与实现细节，语言表达也更符合人类写作习惯。

---

# MLSYS 硬件探针系统实验报告

## 1. 项目目标

系统目标是利用 **NVIDIA Nsight Compute（ncu）** 分析 CUDA kernel 的性能瓶颈（如 Roofline 模型、存储层次、Tensor Core、占用率等），并通过自主生成并运行微基准测试（micro-benchmarks），在可能存在频率限制、SM 掩码或 API 误导的环境下，推断出底层硬件参数，包括 L1/L2/DRAM 延迟、有效峰值带宽、L2 容量、实际 Boost 频率、Shared Memory bank conflict 惩罚等。

输出格式为结构化 JSON 文件，包含各指标的实测数值、方法论说明及证据摘要。

本仓库中的 `src/hw_probe` 实现了上述 **Agent System**：基于 LangGraph 编排多角色 LLM 流程，在工作区内完成规划、编码、编译、运行与 ncu 数据采集，最终汇总为单一的 `output.json`。

---

## 2. 系统总体设计

### 2.1 设计原则

- **封闭工作区**：所有文件读写与命令执行均限制在配置的工作区根目录下，通过 `assert_under_workspace` 校验路径，避免越权访问宿主文件系统。
- **配置集中化**：运行时参数由 `AppSettings`（Pydantic Settings）统一从环境变量读取，业务代码中不散落 `os.getenv`。开发模式下可将容器默认路径映射为仓库相对路径（如 `target/target_spec.json`、工作区 `.`）。
- **可观测与可审计**：LLM 调用通过 JSONL trace 与 Markdown 会话日志记录；工作区内追加 `results.log`；最终 `output.json` 包含 `results`、`methodology` 以及截断后的 `evidence`，便于评测或 LLM-as-a-Judge 引用。
- **受控执行与安全边界**：`run_shell` 对危险子串（包管理器安装、破坏性命令等）做静态拒绝；子进程使用可中断的 `interruptible_communicate`，配合全局 shutdown 事件，使 Ctrl+C / SIGTERM 能够尽快打断长时间编译或 profiling。

### 2.2 数据流概览

1. **入口 `main.py`**：解析 CLI（如 `--dev`）、启动日志与 trace、加载 `target_spec.json`、可选地将模板 `templates/gemm_stub.cu` 拷贝至 `probes/kernel.cu` 作为首轮冒烟测试。
2. **图执行 `run_probe_graph`**：以初始状态（`targets`、`programmer_rounds=0`、`evidence_log=[]`、会话开始时间等）调用 LangGraph，递归上限 200，**最大并发 1**（工具串行执行，便于中断与推理稳定）。
3. **输出 `write_output_json`**：将图终态中的 `results`、`methodology` 与证据摘要写入工作区的 `output.json`（文件名可配置）。

---

## 3. 智能体架构

系统采用 **“主图多节点 + 单节点内嵌 ReAct 子图”** 的层次结构，实现在 `agents/graph.py` 与 `agents/nodes.py` 中。

### 3.1 拓扑结构（LangGraph）

- **节点**：`supervisor`（监督路由）、`planner`（规划）、`programmer`（编程与执行）、`synthesize`（汇总）。
- **边**：
  - `START → supervisor`；
  - `supervisor` 根据状态字段 `_route` 条件跳转至 `planner` / `programmer` / `synthesize`；
  - `planner`、`programmer` 执行后均返回 `supervisor` 形成闭环；
  - `synthesize → END`。

该结构实现了 **“计划—执行—再决策—汇总”** 的迭代式实验流程，而非单次端到端生成。

### 3.2 共享状态 `ProbeState`

定义于 `agents/state.py`，为 `TypedDict`（`total=False`），主要字段包括：

| 字段 | 含义 |
|------|------|
| `targets` | 待探测指标名列表（来自 `target_spec.json`） |
| `plan` | Planner 产出的 Markdown 实验计划 |
| `programmer_rounds` | 编程子智能体已执行轮次 |
| `evidence_log` | 每轮 programmer 的工具轨迹摘要（字符串列表） |
| `results` / `methodology` | 由 Synthesizer 填充的最终结果与方法论说明 |
| `_route` | Supervisor 决定的下一跳：`planner` / `programmer` / `synthesize` |
| `session_started_utc_iso` | 会话开始时间，用于各角色的时间预算计算 |

---

## 4. 多智能体实现方式

本实现中的“多智能体”并非多个独立进程，而是 **同一 LLM 后端、不同系统提示词与工具权限下的多个逻辑角色**，由状态机显式调度。

### 4.1 Planner（规划智能体）

- **职责**：仅输出可执行、可验证的中文实验计划（微基准形态、建议的 ncu metrics、交叉验证思路），**不**编写 CUDA 代码、**不**执行 shell 命令、**不**假设具体 OS/GPU 型号。
- **实现**：单次 `model.invoke`，无工具绑定；用户消息包含 `<targets>` 与 `<time_budget>`（见 `agents/prompts.py`）。
- **产出**：写入状态的 `plan` 字段。

### 4.2 Programmer（编程与执行子智能体）

- **职责**：将编码、探测、编译、运行与 ncu 采集合并为 **单一 ReAct 子智能体**（`create_react_agent` + `ToolNode`），是唯一能够写工作区文件并调用编译/运行/ncu 的角色。
- **工具**：文件系统三件套（读、写、列目录）、`run_shell`、`compile_cuda`、`run_cuda_binary`、`run_ncu_profile`（详见第 5 节）。
- **轮次与证据**：每轮调用前检查总会话时长；超时则跳过本轮并在 `evidence_log` 中记录 `SKIPPED_DEADLINE`。正常结束时将本轮消息历史压缩为证据摘要，追加到 `evidence_log`，并递增 `programmer_rounds`。
- **异常策略**：ReAct 子图未捕获异常时记录 traceback 到证据，**不**让整个主图崩溃；`ToolNode` 开启 `handle_tool_errors=True`，将 nvcc/ncu 等失败转为 `ToolMessage` 供模型继续推理（暴露错误、继续诊断，而非静默吞掉异常）。

### 4.3 Supervisor（监督智能体）

- **职责**：在继续执行 programmer 与进入 synthesize 之间做路由；不直接执行工具。
- **规则混合**：
  - **硬规则**：无 `plan` → 先去 `planner`；总会话时间到或 `programmer_rounds` 达到 `supervisor_max_loops` → 强制进入 `synthesize`。
  - **软规则**：其他情况下，将 `plan` 摘要、证据尾部、轮次等注入用户消息，要求模型**仅输出** JSON：`{"next":"programmer"|"synthesize"}`；解析失败时默认继续 programmer（并记录日志）。

该设计将安全性与可终止性交由代码阈值控制，将“是否值得再采集一轮证据”交由模型判断，符合防循环、交叉验证、时间预算等工程要求。

### 4.4 Synthesizer（汇总智能体）

- **职责**：根据完整的 `evidence_log` 与 `plan`，为每个 `target` 输出数值或 null 的 JSON；禁止虚构证据中不存在的精确数字。
- **实现**：单次 `model.invoke` + `extract_json_object` 解析；`results` 仅保留 `targets` 中出现的键；同时写入固定模板的 `methodology` 字符串，概括多智能体流水线的语义。

---

## 5. 工具层与运行时闭环

| 工具 | 模块 | 作用 |
|------|------|------|
| `read_workspace_file` / `write_workspace_file` / `list_workspace_dir` | `tools/filesystem.py` | 工作区内的只读、只写、列目录 |
| `run_shell` | `tools/shell.py` | 在工作区子目录下执行 shell；策略拒绝安装类命令 |
| `compile_cuda` | `tools/cuda.py` | 使用 `nvcc -O3 -std=c++17` 编译相对路径的源文件 |
| `run_cuda_binary` | `tools/cuda.py` | 运行已编译的 CUDA 二进制 |
| `run_ncu_profile` | `tools/cuda.py` | 执行 `ncu -f --target-processes all --metrics ... --csv` 采集 CSV 数据 |

`profile_with_ncu` 在未指定 metrics 时使用一组默认指标（与 Roofline、内存、Tensor Core、占用率等相关），供 Programmer 按需自定义。

**推荐闭环**（写入 Programmer 系统提示）：环境探测（`uname`、`command -v nvcc/ncu` 等）→ 写入 `probes/*.cu` → `compile_cuda` → `run_cuda_binary` → `run_ncu_profile`；失败时阅读 stderr 并迭代修正。

---

## 6. 配置、输入输出与可观测性

- **关键配置**（`config/settings.py`）：`API_KEY` / `BASE_URL` / `MODEL`、工作区与 `target_spec` 路径、`shell_timeout_sec`、`max_tool_output_chars`、`react_max_steps`、`supervisor_max_loops`、`max_total_runtime_minutes`、日志与 trace 文件名等。
- **输入**：`services/target_spec.py` 读取 JSON，校验 `targets` 为非空字符串列表。
- **输出**：`services/output_writer.py` 写入 `output.json`，结构为 `{ "results", "methodology", "evidence" }`，与语义上的 `results.json` 对应，并扩展了方法论与证据字段以满足调试与评测需求。
- **可观测性**：`observability/llm_trace.py`（JSONL）、`observability/llm_session_markdown.py`（Markdown 会话）、`logging_setup` 分级日志；`main.py` 在会话起止发送自定义 trace 事件（如 `session_start` / `session_end`）。

