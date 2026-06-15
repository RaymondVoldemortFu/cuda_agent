# memxlife

# 第 3 阶段：自动化 LLM 推理运行时

开始日期：2026 年 5 月 26 日  
第一次必需提交：开始日期后 2 周内  
最多提交次数：开始日期后 3 周内提交 2 次

在本阶段，您将构建一个能够自动生成 LLM 推理运行时的 agent。生成的运行时必须从提供的配置和权重中加载 decoder-only 模型，维护请求状态，并高效执行 prefill 和 decode。该运行时将作为黑盒进行评估：我们会将其 logits 与参考实现进行比较以检查正确性，然后使用 serving-style 请求轨迹驱动它，以测量吞吐量和内存行为。

正确推理是一项硬性要求。未通过正确性检查的提交将不会获得吞吐量分数。

---

## 1. 任务

您的任务是实现一个 agent，用于为一个小型类 LLaMA 的 decoder-only 模型生成推理运行时。生成的运行时必须支持：

- 从提供的权重目录加载模型权重
- 根据 `model_config.json` 构造运行时行为
- 对 prompt token 执行 prefill
- 为每个活跃请求的新 token 执行 decode
- 在多次调用之间维护请求状态
- 移除已完成的请求
- 返回与官方参考实现匹配的 logits

您应该设计运行时，使其能够适用于不同的 batch size、prompt length、decode length 和 request order。官方评估轨迹不会提前公开。

---

## 2. 您必须提交的内容

您的提交必须包含：

- `run.sh`
- 您的 agent 实现以及 agent 所需的任何文件

在 `run.sh` 结束后，您的 agent 必须生成：

- `workspace/engine.py`
- `workspace/results.log`

不要将 `workspace/engine.py` 当作手动提交的静态解决方案。它是由您的 agent 生成的输出产物。日志文件不用于评分。它的作用是让您能够在提交后检查失败原因，例如 agent 错误、代码生成错误、编译错误或本地自测失败。

### 提交约定

评估系统会进入提交根目录并运行：

```bash
bash run.sh
```

在 `run.sh` 结束后，评估系统会从同一目录导入：

```text
workspace/engine.py
```

并运行官方正确性和吞吐量测试框架。

您的 `run.sh` 应该调用您的 agent。如果生成的运行时需要自定义扩展、生成文件或本地自测，请在此过程中准备好。评估器不会使用您日志文件中的自报结果；它会直接调用生成的运行时。

---

## 3. 提供的输入

模型配置文件是：

```text
/target/model_config.json
```

在公开 skeleton 中，对应路径是：

```text
target/model_config.json
```

该文件描述模型结构，包括 hidden size、层数、attention head 数量、key-value head 数量、词表大小以及相关参数。您的运行时不应硬编码这些值。它应该根据传递给 `create_engine(...)` 的 `model_config` 参数动态构造 engine。

模型权重目录是：

```text
/target/weights
```

在公开 skeleton 中，权重文件是：

```text
target/weights/model.pt
```

公开 skeleton 使用单个 PyTorch state dict。隐藏评估会通过同一个 `weight_dir` 参数提供权重。

---

## 4. 必需的运行时接口

`workspace/engine.py` 必须定义：

```python
def create_engine(model_config: dict, weight_dir: str, device: str = "cuda"):
    return Engine(...)
```

返回的对象必须支持：

```python
class Engine:
    def prefill(self, request_ids, input_ids):
        ...

    def decode(self, request_ids, token_ids):
        ...

    def remove(self, request_ids):
        ...
```

### `prefill(request_ids, input_ids)`

输入：

- `request_ids`：请求 ID 的列表，例如 `[0, 1, 2]`
- `input_ids`：1D `torch.Tensor` token 序列的列表，每个请求对应一个序列

输出：

- 一个形状为 `[batch_size, vocab_size]` 的 logits tensor
- 第 `i` 行必须包含 `request_ids[i]` 的 last-token logits

对某个请求调用 `prefill(...)` 应该创建或替换该请求的状态。它不应该清除无关请求的状态。

### `decode(request_ids, token_ids)`

输入：

- `request_ids`：已有请求 ID 的列表
- `token_ids`：形状为 `[batch_size]` 的 1D `torch.Tensor`，每个请求对应一个新 token

输出：

- 一个形状为 `[batch_size, vocab_size]` 的 logits tensor
- 第 `i` 行必须包含将 `token_ids[i]` 追加到 `request_ids[i]` 之后的 last-token logits

### `remove(request_ids)`

输入：

- `request_ids`：已完成请求 ID 的列表

该方法不需要返回任何内容。它应该释放或删除与这些 ID 相关联的请求状态。

---

## 5. 正确性检查

官方评估器会使用具有相同隐藏模型配置和权重的 PyTorch 参考实现。我们比较的是 logits，而不是生成文本。

正确性使用以下方式检查：

```math
|y_{\mathrm{student}} - y_{\mathrm{ref}}| \leq \mathrm{atol} + \mathrm{rtol} \cdot |y_{\mathrm{ref}}|
```

公开 skeleton 使用：

```math
\mathrm{atol}=10^{-2}, \quad \mathrm{rtol}=10^{-2}
```

公开正确性测试使用：

```python
torch.allclose(student_logits, ref_logits, atol=1e-2, rtol=1e-2)
```

正确性测试覆盖：

- 单请求 prefill
- 单请求 decode
- 多请求 prefill
- 多请求 decode
- 插入新请求
- 移除请求并继续 decode 其他请求

如果某个 case 未通过正确性，该 case 不会获得吞吐量分数。

---

## 6. 吞吐量评估

官方评估器会直接驱动您的 engine：

```python
engine = create_engine(model_config, weight_dir, device)
engine.prefill(...)
engine.decode(...)
engine.remove(...)
```

计时区域包括对以下内容的调用：

- `prefill(...)`
- `decode(...)`
- `remove(...)`

计时区域不包括 `create_engine(...)` 或初始权重加载。如果您在被计时的调用中执行 lazy compilation 或昂贵初始化，这部分时间会被计入。

吞吐量报告为：

```math
\mathrm{tokens/s}=\frac{\mathrm{prefill\ tokens}+\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}
```

Decode 吞吐量报告为：

```math
\mathrm{decode\ tokens/s}=\frac{\mathrm{decode\ tokens}}{\mathrm{elapsed\ seconds}}
```

公开 benchmark 包括三类 case：

- `prefill`：batched long-prompt prefill
- `decode`：多个活跃请求执行重复 decode step
- `mixed`：包含 prefill、decode 和 remove 操作的 serving-style trace

隐藏评估将使用相同的接口和评估方式，但会使用隐藏的模型大小、权重、batch size、prompt length、decode step 和 request trace。

---

## 7. 评分策略

正确性是一项硬性要求。

未通过正确性检查的提交将不会获得吞吐量分数。

对于通过正确性的提交，最终分数为：

- 70% 吞吐量
- 30% Agent 实现 / 工程方法

### 吞吐量

吞吐量评分基于官方 benchmark trace。评估器会在适当情况下使用 warmup、重复测量和 median timing。

benchmark 会考虑 prefill、decode 和 mixed serving 行为。您应该优化 engine 的整体运行时行为，而不只是优化某一种孤立调用模式。

### Agent 实现 / 工程方法

这一部分奖励体现真实工程流程的提交，包括以下因素：

- 清晰的运行时组织
- 与参考实现对照的本地正确性测试
- 用于决策的 benchmark 和 profiling
- 迭代改进
- 对不同 model config 和 request pattern 的鲁棒处理
- 通过 `run.sh` 和日志实现可复现性

该项目并不是要求一个只适用于公开 toy case 的手写静态解决方案。一个强提交应该使用公开输入来验证接口，然后构建一个能够泛化到隐藏 case 的运行时。

---

## 8. 允许的优化方向

您可以使用如下技术优化运行时：

- 真正的 per-layer KV cache
- batched prefill 和 decode
- PyTorch SDPA 或其他 PyTorch primitive
- Triton kernel
- C++/CUDA 扩展
- 用于 RMSNorm、RoPE、attention、MLP 或 cache 操作的自定义 kernel
- 更好的内存布局和请求状态管理

您应该避免依赖完整推理框架作为最终运行时实现。评估器期望您的 `engine.py` 直接实现所需接口。

---

## 9. 公开 Skeleton

如果公开权重文件缺失，请使用以下命令重新生成：

```bash
python3 scripts/generate_toy_weights.py \
  --config target/model_config.json \
  --output target/weights/model.pt
```

运行公开正确性测试：

```bash
python3 evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

运行公开吞吐量 benchmark：

```bash
python3 evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

或者同时运行两者：

```bash
bash scripts/run_public_tests.sh
```

如果您的默认 `python3` 没有 PyTorch，请指定 Python 解释器：

```bash
PYTHON=/path/to/python-with-torch bash scripts/run_public_tests.sh
```

---

## 10. Baseline

公开 skeleton 已经在 `workspace/engine.py` 中包含了一个示例生成产物，使您可以立即运行评估器。该文件是一个最小 PyTorch baseline。它为每个请求存储完整 token 序列，并在每次 decode 调用时重新计算完整序列。这很慢，但展示了所需接口和正确请求语义。在您自己的提交中，您的 agent 必须在 `run.sh` 启动后生成 `workspace/engine.py`。

重要优化方向包括：

- 实现真正的 per-layer KV cache
- 让 `decode(...)` 只计算新 token
- 在请求之间 batch 工作
- 减少 Python overhead
- 优化 attention、MLP、RMSNorm、RoPE 和 cache 操作
- 根据 `model_config.json` 适配实现选择

---

## 11. 总结

在本项目中，您正在构建一个能够自动为 decoder-only 语言模型生成推理运行时的 agent。

您的提交应该：

- 提供 `run.sh`
- 从 `run.sh` 调用您的 agent
- 生成 `workspace/engine.py`
- 实现 `create_engine(...)`
- 支持 `prefill(...)`、`decode(...)` 和 `remove(...)`
- 使用 request ID 维护独立请求状态
- 匹配参考 logits
- 在 serving-style trace 上优化吞吐量

只有正确实现才能获得吞吐量分数。在正确提交中，评分基于：

- 70% 吞吐量
- 30% agent 实现 / 工程方法
