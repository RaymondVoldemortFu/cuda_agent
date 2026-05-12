# MLSYS 课程项目

随着 Agent 时代的到来，系统工程的格局正在经历一次根本性的范式转变。传统上，算子优化、高性能 GPU 调优这类任务只属于极少数顶尖专家的领域；他们需要具备深厚而垂直的知识体系，既要理解高层算法，也要掌握 GPU 硬件架构的复杂细节。过去，要大规模完成这种复杂度的项目，往往需要数千人的团队。

而今天，这道门槛正在消失。我们正在进入这样一个时代：单个个体借助智能体，就能够完成过去需要整个部门才能完成的工作。通过设计能够感知硬件行为、推理性能瓶颈并迭代改进代码的系统，我们可以将 AI 基础设施中的“专家层”自动化。这正是本学期项目的核心目标：从“手动编写代码”转向“设计能够自主编写、评估并优化基础设施的自治系统”——其覆盖范围包括 GPU profiling、CUDA kernel 自动调优，以及自动化 LLM 基础设施生成。

---

# 第一阶段：GPU 性能分析指南：使用 ncu 识别瓶颈

**截止时间：2026 年 4 月 21 日上午 8 点**

NVIDIA Nsight Compute（ncu）是一个面向 CUDA kernel 的交互式内核级性能分析工具。不同于观察全局时间线的 Nsight Systems（nsys），ncu 会深入到每个 kernel 的内部执行过程，展示硬件资源究竟是如何被消耗的。

在本项目中，您的 Agent 需要分析 ncu 输出的各类指标，以判断某个算子（例如矩阵乘法）的性能瓶颈。下面列出了核心指标类别及其在性能调优中的意义。

---

## 1.1 核心概览指标

在进行详细分析之前，首先使用 Roofline Model（屋顶线模型）判断该算子是 **Compute-Bound（计算受限）** 还是 **Memory-Bound（内存受限）**。

### sm__throughput.avg.pct_of_peak_sustained_elapsed（计算利用率）

**含义：** SM（Streaming Multiprocessor，流式多处理器）计算单元的利用率。
**分析：** 如果该值较高，说明 GPU 的计算核心已经被充分占满。

### gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed（内存吞吐率）

**含义：** GPU 内存带宽的利用率。
**分析：** 如果该值较高，说明数据传输速度正在限制性能。

### SOL（Speed of Light）

ncu 报告中经常提到的 “SOL Compute” 和 “SOL Memory”，表示当前算子达到了硬件理论峰值性能的百分之多少。

---

## 1.2 存储层次结构指标

如果该算子是内存密集型的，那么您需要判断存储层次结构中的哪一层是瓶颈。

### l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum（L1 Cache）

**分析：** 反映 L1 Cache 的命中率和吞吐情况。频繁访问全局内存且无法命中 L1 Cache，会显著增加延迟。

### l2__throughput.avg.pct_of_peak_sustained_elapsed（L2 Cache）

**分析：** L2 是显存（VRAM）与 SM 之间的桥梁。若 L2 利用率极高，则说明这两层之间存在大量数据交换。

### dram__throughput.avg.pct_of_peak_sustained_elapsed（VRAM/DRAM）

**分析：** 表示外部显存带宽利用率。如果该值超过 80%，说明您的代码很可能需要更好的数据复用方式（例如使用 Shared Memory）。

---

## 1.3 计算单元指标

如果算子是计算受限的，那么您需要进一步判断究竟是哪类计算单元在工作。

### sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active（Tensor Cores）

**分析：** 这一项非常关键。对于深度学习中的矩阵乘法，如果该指标较低，就说明您没有有效利用 Tensor Cores。

### sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active（FP32/FMA）

**分析：** 传统单精度浮点单元的利用率。

### sm__sass_thread_inst_executed_op_fp32_pred_on.sum

**分析：** 实际执行的 FP32 指令总数。

---

## 1.4 占用率与调度

有时硬件利用率较低，并不是因为算力不足，而是因为线程没有成功“填满”GPU。

### sm__maximum_warps_per_active_cycle_pct（理论占用率）

**含义：** 基于寄存器和共享内存使用情况，在硬件理论上可同时运行的 warp 百分比上限。

### sm__warps_active.avg.pct_of_peak_sustained_active（实际占用率）

**含义：** 算子执行过程中实际达到的占用率。

**分析：** 如果实际占用率明显低于理论占用率，则说明可能存在指令延迟问题，或者 thread block 分布不均匀。

---

## 1.5 常见瓶颈与诊断表

| 瓶颈类型                      | 关键指标                                                 | 优化方向                                |
| ------------------------- | ---------------------------------------------------- | ----------------------------------- |
| VRAM Bound（显存受限）          | dram__throughput > 70%                               | 减少内存访问；增加数据复用；使用 Shared Memory      |
| Compute Bound（计算受限）       | 高 sm__throughput，高 tensor_op_hmma                    | 考虑算法改进，或降低精度（如 FP32 改为 FP16/BF16）   |
| Uncoalesced Access（非合并访问） | l1tex__t_sectors_pipe_lsu_mem_global_op_ld 过高        | 检查内存访问模式；确保相邻线程访问相邻地址               |
| Warp Divergence（warp 分歧）  | sm__sass_thread_inst_executed_per_inst_executed < 32 | 减少 if/else 分支；确保同一 warp 内线程遵循相同执行路径 |
| Bank Conflict（bank 冲突）    | l1tex__data_bank_conflicts_pipe_lsu.sum > 0          | 调整 Shared Memory 索引方式（例如使用 Padding） |

---

## 1.6 给学生的建议：如何让 Agent 使用这些数据？

### 第一步：获取 Roofline

先让 Agent 读取 `sm__throughput` 和 `gpu__compute_memory_throughput`。

### 第二步：进行特征判断

* 如果 **Memory % > Compute %**，就继续深入分析 `dram` 和 `l1/l2` 指标。
* 如果 **Compute % > Memory %**，就检查是否触发了 `tensor_op`。

### 第三步：寻找异常

检查占用率是否过低，或者是否存在 Bank Conflict。

### 第四步：映射回代码

将这些指标与您的 CUDA kernel 代码关联起来分析。
例如：循环展开是否不足？是否缺少 `__shared__` 数组？

### 如何通过命令行获取这些指标？

```bash
# 示例：获取一个矩阵乘法 kernel 的所有详细指标
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active ./your_executable
```

---

## 1.7 硬件内在特性分析：探测 GPU 的“DNA”

在项目的这一高级阶段中，您的 Agent 不再只是被动报告已有 kernel 的性能。它必须充当一个 **Hardware Probe（硬件探针）**。目标是通过自主生成并分析微基准测试（micro-benchmarks），来“逆向推断”底层 GPU 的物理特性和架构极限。

---

### 1.7.1 探测目标（Target Metrics）

您的 Agent 将被要求识别以下硬件内在参数：

#### Memory Latency Hierarchy（内存延迟层级）

精确测量 L1 Cache、L2 Cache 和 DRAM 的访问周期。
这要求 Agent 生成 “Pointer Chasing” 类型的 kernel，以绕过硬件预取器。

#### Effective Peak Bandwidth（有效峰值带宽）

确定在当前条件下 Shared Memory 与 Global Memory（VRAM）所能达到的最大实际吞吐率。

#### L2 Cache Capacity（L2 缓存容量）

识别延迟—容量曲线中的“悬崖点（cliff）”，从而确定 L2 Cache 的精确物理大小。

#### Actual Boost Frequency（实际加速频率）

报告 GPU 在持续计算负载下稳定运行时的核心频率（MHz）。

#### Resource Penalties（资源惩罚）

量化 Shared Memory 中发生 bank conflict 相较于无冲突访问所带来的额外延迟代价。

---

### 1.7.2 提交与评测流程

#### 学生提交内容

您必须提交您的 Agent System（基础设施代码）。

#### 输入

评测系统会提供一个 `target_spec.json`，其中包含需要识别的硬件指标列表。

**输入示例：**

```json
{"targets": ["dram_latency_cycles", "max_shmem_per_block_kb", "actual_boost_clock_mhz"]}
```

#### 输出

您的 Agent 必须输出一个 `results.json`，其中包含识别出的数值结果。

**输出示例：**

```json
{"dram_latency_cycles": 442, "max_shmem_per_block_kb": 48, ...}
```

#### Ground Truth Comparison（与真实值对比）

您的 Agent 输出结果将与服务器端参考基准程序生成的高精度真实值进行比较。

---

### 1.7.3 反作弊与环境变化

为了确保您的 Agent 进行的是真正的硬件分析，而不是简单地做“查表”（例如搜索 “A100 specs”），评测环境会被动态修改：

#### Non-Standard Frequency Locking（非标准频率锁定）

可能会使用 `nvidia-smi` 将 GPU 的核心频率和显存频率锁定到任意、非标准的频率值（例如 825 MHz，而不是 1410 MHz）。
此时，静态查表将给出错误的带宽或 GFLOPS 结果。

#### Resource Masking（资源遮罩 / SM 限制）

系统可能会通过 CUDA 环境变量，将 kernel 的执行限制在部分 SM 上，或者限制每个 block 可用的内存。

#### Instruction Set Restrictions（指令集限制）

类似 `cudaGetDeviceProperties` 这类标准 API 调用，可能会被拦截，或者返回经过虚拟化/误导性处理的数据。

#### 建议

您的 Agent 应采用 **Multi-Strategy Fusion（多策略融合）** 方法：
将底层微基准测试（编写小型 C++/CUDA 探测程序）、二进制执行以及 ncu 指标分析结合起来，并通过交叉验证确认结果。

---

### 1.7.4 基于 LLM 的评估与评分

为了对数值精度和工程推理能力进行更全面的评估，本项目采用 **LLM-as-a-Judge** 框架进行评分。

#### 评分流程

提交后，评测系统会将以下三部分内容输入一个高能力大型语言模型（例如 Gemini 3.1 Pro）：

1. **Student Agent Output**
   您的 Agent 最终输出的 `results.json`，以及 Agent 提供的推理过程/日志。

2. **Ground Truth Data**
   在特定环境下（包括任何激活的频率锁定或资源遮罩）通过参考基准程序测得的精确硬件参数。

3. **Experimental Evidence**
   Agent 执行过程中生成的 ncu trace 与微基准测试摘要。

---

## 评分标准（总分 100 分）

### 1. Numerical Alignment（数值一致性，70 分）

LLM 会将您 Agent 报告的数值与 Ground Truth 进行比较。

* **满分：** 数值落在可接受的工程误差范围内（例如延迟 ±5%，带宽 ±2%）。
* **部分得分：** 数值趋势正确，但存在校准误差。
* **零分：** 数值与标准“在线规格表”一致，却无法反映测试中实际提供的限频/遮罩环境。

### 2. Engineering Reasoning & Methodology（工程推理与方法论，30 分）

LLM 会评估您的 Agent 是“如何”以及“为什么”得出这些结论的。

* **Inference Quality（推断质量）：**
  Agent 是否正确识别 GPU 被锁定在非标准频率下？是否注意到 SM 遮罩？

* **Micro-benchmark Validity（微基准测试有效性）：**
  Agent 是否生成了合适的 CUDA kernel（例如用于测延迟的 pointer-chasing），还是依赖了可能受限的 API 调用？

* **Cross-Verification（交叉验证）：**
  Agent 是否进行了多轮实验，或使用不同类型的探针来确认可疑指标？

