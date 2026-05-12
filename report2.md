# MLSYS LoRA 优化 Agent 系统实验报告

## 1. 项目目标

系统目标是为 LoRA 风格算子

```text
Y = W X + A(B^T X)
```

构建一个可迭代优化的 CUDA Agent。输入张量均为 `float32`，其中 `W` 与 `X` 的形状为 `[d, d]`，`A` 与 `B` 的形状为 `[d, 16]`，隐藏维度 `d` 位于 `[3584, 4608]`。系统需要在运行时生成、编译、校验并 benchmark 多个候选 CUDA/PyTorch extension 实现，最终将当前最优且可编译的实现维护在工作区根目录的 `optimized_lora.cu`。

本仓库中的 `src/hw_probe` 复用了第一阶段的 **LangGraph 多角色 Agent 架构**，但将任务语义从硬件探针改造为 LoRA 算子优化。整个流程由 Planner 制定搜索策略、Programmer 生成和评测候选、Supervisor 控制迭代终止、Synthesizer 汇总最终结果，形成可审计的优化闭环。

---

## 2. 系统总体设计

### 2.1 设计原则

- **始终保持可交付 CUDA 文件**：`main.py` 在进入优化图之前调用 `seed_initial_optimized_lora`，若工作区尚无 `optimized_lora.cu`，立即写入一个基于 ATen/cuBLAS 的正确 baseline，避免超时或中断时没有可编译产物。
- **真实候选搜索**：Programmer 不直接覆盖最终文件，而是将候选写入 `stage2_candidates/*.cu`，再调用 `evaluate_lora_candidate` 完成编译、正确性校验、benchmark 与择优提升。
- **正确性优先**：候选必须通过与 PyTorch reference 一致的 `torch.allclose(rtol=1e-4, atol=1e-4)` 校验，错误候选只记录失败信息，不允许替换当前 best。
- **覆盖边界形状**：评测默认包含 `3584, 3601, 4096`，其中 `3601` 专门用于暴露向量化 kernel 在非 4 对齐维度上的 tail、对齐与越界问题。
- **复用系统 PyTorch/CUDA**：`run.sh` 使用系统 `python3` 运行 agent，使官方环境中预装的 PyTorch、CUDA toolkit 与 extension toolchain 可直接参与编译和 benchmark。

### 2.2 数据流概览

1. **入口 `run.sh`**：进入提交根目录，检查 LangGraph、LangChain、Pydantic Settings 与 `ninja` 是否可用；缺失时通过 pip 安装 agent 侧依赖，然后执行 `python3 -m hw_probe.main`。
2. **启动 `main.py`**：加载 `API_KEY` / `BASE_URL` / `BASE_MODEL` 等配置，初始化日志与 trace，写入 baseline `optimized_lora.cu`，再调用 `run_probe_graph`。
3. **图执行 `run_probe_graph`**：沿用 `supervisor → planner/programmer/synthesize` 的状态机结构，通过 `ProbeState` 传递计划、轮次、证据与最终结果。
4. **候选评测 `evaluate_lora_candidate`**：编译指定 `.cu` 文件，生成 synthetic FP32 LoRA 输入，对比 PyTorch reference，使用 CUDA event 记录 median latency，并按分数决定是否提升为根目录 `optimized_lora.cu`。
5. **输出 `write_output_json`**：图结束后将 `results`、`methodology` 与最近证据写入 `output.json`；候选明细写入 `stage2_candidates/*.json`，当前 best 元信息写入 `stage2_best.json`。

---

## 3. 智能体架构

系统继续采用 **“主图多节点 + Programmer 内嵌 ReAct 子图”** 的层次结构，核心实现在 `agents/graph.py` 与 `agents/nodes.py` 中。

### 3.1 拓扑结构（LangGraph）

- **节点**：`supervisor`（监督路由）、`planner`（优化计划）、`programmer`（候选生成与执行）、`synthesize`（结果汇总）。
- **边**：
  - `START → supervisor`；
  - `supervisor` 根据 `_route` 跳转至 `planner` / `programmer` / `synthesize`；
  - `planner` 和 `programmer` 执行后回到 `supervisor`；
  - `synthesize → END`。

该结构保留了第一阶段“计划—执行—再决策—汇总”的迭代机制，但每轮执行的目标从采集硬件指标变为产生新的 LoRA CUDA 候选并用实测结果驱动下一步优化。

### 3.2 共享状态 `ProbeState`

`ProbeState` 仍然是图节点之间的共享状态容器，主要字段包括：

| 字段 | 含义 |
|------|------|
| `targets` | 当前优化目标描述，即 LoRA forward 算子 |
| `plan` | Planner 产出的搜索计划 |
| `programmer_rounds` | Programmer 已执行轮次 |
| `evidence_log` | 每轮 ReAct 工具调用与结果摘要 |
| `results` / `methodology` | Synthesizer 汇总出的最终候选、分数、形状覆盖与方法说明 |
| `_route` | Supervisor 决定的下一跳 |
| `session_started_utc_iso` | 会话开始时间，用于控制 30 分钟预算 |

---

## 4. 多智能体实现方式

本实现中的多智能体仍是同一 LLM 后端下的多个逻辑角色，各角色通过不同系统提示词、工具权限和状态字段协同完成优化搜索。

### 4.1 Planner（规划智能体）

- **职责**：根据 LoRA 算子、形状范围与时间预算制定候选搜索策略。
- **约束**：Planner 不写 CUDA 代码，不调用工具；它只描述候选顺序、评测 shape、择优规则和收尾策略。
- **策略倾向**：优先建立基于 ATen/cuBLAS 的正确 baseline，再优化低秩项 `A(B^T X)` 的加法路径，避免在短时间内从零实现完整 FP32 GEMM。

### 4.2 Programmer（编程与执行子智能体）

- **职责**：作为唯一能写文件和调用执行工具的 ReAct 子智能体，负责生成候选 `.cu`、调用评测工具、阅读结果并继续迭代。
- **工具**：除工作区文件工具与 `run_shell` 外，新增 `evaluate_lora_candidate`，用于 LoRA 专用编译、校验、benchmark 和 best 提升。
- **候选管理**：候选写入 `stage2_candidates/<name>.cu`；根目录 `optimized_lora.cu` 只能由评测工具在候选正确且满足择优条件时原子替换。
- **异常策略**：编译错误、CUDA runtime 错误、correctness 失败都会进入工具返回 JSON 与候选记录文件，供后续推理使用，不通过默认值或静默跳过掩盖。

### 4.3 Supervisor（监督智能体）

- **职责**：决定继续生成候选还是进入汇总。
- **硬规则**：无计划时先进入 Planner；达到总时间预算或轮次上限时进入 Synthesizer。
- **软规则**：若已有 promoted 候选且继续收益有限，或最近证据显示重复失败，则收尾；若尚无任何 `evaluate_lora_candidate` 证据，则继续执行 Programmer。

### 4.4 Synthesizer（汇总智能体）

- **职责**：从 `evidence_log` 中汇总最佳候选、平均 speedup、已验证 shape、是否 promoted，以及剩余风险。
- **输出格式**：只输出 JSON 对象，至少包含 `best_candidate`、`best_score`、`promoted`、`tested_shapes`、`summary` 与 `remaining_risk`。
- **证据约束**：只允许引用工具结果中出现过的数值；没有证据的候选不能被描述为已通过或已提升。

---

## 5. 工具层与优化闭环

| 工具 | 模块 | 作用 |
|------|------|------|
| `read_workspace_file` / `write_workspace_file` / `list_workspace_dir` | `tools/filesystem.py` | 在工作区内读取、写入和列举候选文件 |
| `run_shell` | `tools/shell.py` | 轻量环境探测或查看工作区状态 |
| `compile_cuda` / `run_cuda_binary` / `run_ncu_profile` | `tools/cuda.py` | 保留第一阶段 CUDA 探针能力，必要时可辅助诊断 |
| `evaluate_lora_candidate` | `tools/lora.py` | 编译 PyTorch extension、生成 synthetic 输入、检查 correctness、benchmark、提升 best |

`evaluate_lora_candidate` 是第二阶段的核心工具。它接收候选源码相对路径、候选名、测试 shape、warmup 与 benchmark 迭代次数，内部执行：

1. 使用 `torch.utils.cpp_extension.load` 编译候选 `.cu`。
2. 为每个 `d` 生成 deterministic synthetic FP32 输入 `W, X, A, B`。
3. 计算 `module.forward(W, X, A, B)` 与 reference `W @ X + A @ (B.T.contiguous() @ X)`。
4. 记录 `max_abs_err`、`rel_l2_err` 与 `torch.allclose` 结果。
5. 对通过 correctness 的候选使用 CUDA event 统计 median runtime。
6. 计算相对 PyTorch reference 的 speedup，并在候选优于当前 best 或覆盖了更安全 shape 时原子更新 `optimized_lora.cu`。

---

## 6. CUDA 实现策略

最终候选采用混合策略：保留 ATen/cuBLAS 处理大矩阵乘法，将自定义 CUDA kernel 聚焦在低秩项加法上。

```text
Y = W @ X
T = B^T @ X
Y += A @ T
```

其中 `W @ X` 和 `B^T @ X` 仍由 PyTorch/ATen 调用底层高性能 GEMM；自定义 kernel 针对 `r=16` 的固定低秩维度展开累加，将 `A @ T` 直接加到 `Y` 中，减少额外中间矩阵与通用 GEMM 调度开销。

当前最优实现的关键点包括：

- **固定 rank 展开**：`r=16` 在 kernel 内用 `#pragma unroll` 展开，降低循环控制开销。
- **向量化列处理**：当 `d % 4 == 0` 时，每个线程使用 `float4` 处理四个连续列元素，提高 load/store 粒度。
- **非对齐安全路径**：当 `d` 不是 4 的倍数时，禁止 `float4` 访问，切换到标量路径处理 tail，避免 misaligned address 与越界。
- **原地累加**：kernel 直接对 `Y` 做 `+=`，保持输出接口简单，并避免额外的张量加法 kernel。

---

## 7. 配置、输入输出与可观测性

- **配置来源**：`AppSettings` 从环境变量读取 `API_KEY`、`BASE_URL`、`BASE_MODEL`、workspace、时间预算、ReAct 步数、日志文件名等配置。
- **默认工作区**：生产模式下为 `/workspace`；开发模式 `--dev` 会映射到仓库根目录，便于本地测试。
- **主交付物**：`optimized_lora.cu` 位于工作区根目录，由 `main.py` 先写 baseline，再由 `evaluate_lora_candidate` 按实测结果提升。
- **运行报告**：`output.json` 由 `write_output_json` 写入，包含 `results`、`methodology` 与截断证据。
- **候选记录**：每次工具评测都会在 `stage2_candidates/*.json` 写入完整候选结果；当前 best 写入 `stage2_best.json`。
- **可观测性**：LLM 调用链写入 `log/llm_trace.jsonl`，人类可读对话写入 `log/llm_session.md`，调试日志写入 `log/hw_probe.debug.log`，运行摘要追加到 `results.log`。

该设计使最终 CUDA 文件、候选演化过程和 LLM 决策证据都能独立检查；即使优化过程中出现失败候选，也会以结构化方式暴露错误原因，而不会影响已验证 best 的可用性。
