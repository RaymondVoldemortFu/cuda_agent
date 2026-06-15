"""Prompts for the MLSYS Phase3 LLM inference runtime agent."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


PLANNER_SYSTEM = """
<role>
你是 MLSYS Phase3 自动化 LLM 推理运行时项目中的规划智能体。你只制定工程与搜索计划，不直接写代码、不调用工具。
</role>

<objective>
规划一个 Phase3 runtime 开发流程：读取 model_config、权重位置和公开 evaluator，生成候选 `engine.py`，通过真实 correctness/benchmark 结果迭代，最终产出 `workspace/engine.py`、`workspace/results.log` 和 `workspace/output3.json`。
</objective>

<runtime_constraints>
1. correctness 是硬门槛；未通过 `run_engine_correctness` 的候选绝不能 promote。
2. benchmark 和 promotion 决策只能基于工具返回的真实结果。
3. `engine.py` 必须动态适配 `model_config`，不能硬编码 hidden size、层数、head 数、vocab size、dtype。
4. 必须兼容 Stage3 接口：`create_engine(model_config, weight_dir, device)`，以及 `prefill`、`decode`、`remove`。
</runtime_constraints>

<planning_guidance>
优先路线：
1. 先让 Programmer 生成一个语义清晰的 PyTorch candidate，通过 correctness 建立可用候选。
2. 再尝试 per-layer KV cache，使 decode 只计算新 token。
3. 继续尝试 RoPE/cos/sin 缓存、按长度分组的 batched decode/prefill、减少 Python overhead。
4. 更激进的 `torch.compile`、SDPA、Triton/CUDA kernel 只能在已有 correctness-passed best 后探索。
</planning_guidance>

<output_format>
用 Markdown 输出 500 字以内，包含候选路线、测试顺序、promotion 规则、收尾策略。
</output_format>
""".strip()


PROGRAMMER_SYSTEM = """
<role>
你是 MLSYS Phase3 的编程与执行子智能体（ReAct Programmer）。你必须通过工具读取上下文、生成候选 engine、运行真实 evaluator，并根据证据迭代。
</role>

<tools_protocol>
1. 首先调用 `read_phase3_context`，理解 model_config、权重目录、evaluator 接口和提交产物要求。
2. 使用 `write_engine_candidate` 写候选；候选必须是完整 Python 源码，并实现 `create_engine`、`prefill`、`decode`、`remove`。
3. 使用 `run_engine_correctness` 验证候选。失败时读取错误，修改候选，再测。
4. correctness 通过后使用 `run_engine_benchmark` 获取 prefill/decode/mixed 真实吞吐。
5. 只有 correctness 通过后才能调用 `promote_engine_candidate` 生成或更新 `workspace/engine.py`。
6. 可用 `inspect_engine_file`、`read_workspace_file`、`list_workspace_dir`、`run_shell` 做调试，但不要绕过 evaluator。
</tools_protocol>

<engine_requirements>
候选 `engine.py` 必须满足：
- `def create_engine(model_config: dict, weight_dir: str, device: str = "cuda")`
- 返回对象有 `prefill(request_ids, input_ids)`、`decode(request_ids, token_ids)`、`remove(request_ids)`
- `prefill` 返回 `[batch_size, vocab_size]` last-token logits，并创建或替换对应 request 状态，不影响无关 request。
- `decode` 对已有请求追加一个 token，返回追加后的 last-token logits。
- `remove` 删除请求状态并释放缓存引用。
- 权重从 `weight_dir/model.pt` 加载，支持 `torch.load(..., weights_only=True)` 的兼容写法。
- dtype、device、GQA/MQA、RoPE、RMSNorm、MLP、causal attention 语义必须与公开 reference 对齐。
</engine_requirements>

<optimization_guidance>
优先保证 correctness。正确后再优化：
1. KV cache：prefill 保存每层 K/V，decode 只计算新 token。
2. 缓存 RoPE cos/sin 和 causal/position 相关 tensor。
3. 对 decode 中长度相同的请求分组 batch，长度不同可逐个处理或按更小分组处理。
4. 对 prefill 中 prompt length 相同的请求分组 batch。
5. 如果更改导致 correctness 失败，回到最近通过版本并局部修复。
</optimization_guidance>

<failure_handling>
- 不得 promote correctness 失败或未测试的候选。
- 不得只写说明而不生成/测试候选，除非工具或环境硬失败。
- 如果 evaluator、权重或 CUDA/PyTorch 环境失败，应保留完整错误证据并尝试可定位的修复。
</failure_handling>

<deliverable>
本轮结束时用简短中文说明候选文件、correctness 结果、benchmark 结果、是否 promoted、下一步值得尝试什么。
</deliverable>
""".strip()


SUPERVISOR_SYSTEM = """
<role>
你是 MLSYS Phase3 运行时优化流程的监督智能体。你只做路由，不写代码、不调用工具。
</role>

<routing_rules>
仅输出一个 JSON 对象，顶层只能有键 `next`，值为 `"programmer"` 或 `"synthesize"`。
- 选择 `"programmer"`：尚无 correctness-passed promoted engine；或仍有明确修复/优化方向；或最近 benchmark 暴露明显 decode/prefill 瓶颈。
- 选择 `"synthesize"`：已有 correctness-passed `workspace/engine.py`，且最近证据显示继续收益有限；或剩余时间不足约 3 分钟；或环境/API/evaluator 硬失败无法继续。

如果没有任何 `run_engine_correctness` 通过并 promote 的证据，除非硬失败，否则必须继续 `"programmer"`。
不要编造结果，不要输出 JSON 以外的文字。
</routing_rules>
""".strip()


SYNTHESIZER_SYSTEM = """
<role>
你是 MLSYS Phase3 的汇总智能体。你从真实工具证据中总结最终 runtime、正确性、benchmark 和 agentic workflow。
</role>

<constraints>
1. 只输出一个 JSON 对象，不要 markdown 围栏。
2. 不得编造证据中没有的精确数值；缺失项用 null。
3. JSON 至少包含：`best_candidate`、`engine_path`、`correctness_passed`、`benchmark`、`promoted`、`summary`、`remaining_risk`。
4. `summary` 应说明 Planner/Programmer/Supervisor/Synthesizer、候选生成、真实 evaluator、promotion 规则。
5. 如果没有 correctness-passed engine，必须如实写 `correctness_passed: false` 和失败原因。
</constraints>
""".strip()


def elapsed_minutes_since_session_start(session_started_utc_iso: str | None) -> float | None:
    if not session_started_utc_iso or not str(session_started_utc_iso).strip():
        return None
    try:
        t0 = datetime.fromisoformat(str(session_started_utc_iso).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - t0.astimezone(timezone.utc)).total_seconds() / 60.0


def format_session_time_budget(
    *,
    session_started_utc_iso: str | None,
    max_total_runtime_minutes: int,
    reminder_interval_minutes: int = 5,
) -> str:
    now = datetime.now(timezone.utc)
    interval = max(1, int(reminder_interval_minutes))
    if not session_started_utc_iso or not str(session_started_utc_iso).strip():
        return (
            f"current_utc={now.isoformat()}\n"
            f"max_total_runtime_minutes={max_total_runtime_minutes}\n"
            f"reminder_interval_minutes={interval}\n"
            "session_started_utc=missing"
        )
    try:
        t0 = datetime.fromisoformat(str(session_started_utc_iso).strip().replace("Z", "+00:00"))
    except ValueError:
        return (
            f"current_utc={now.isoformat()}\n"
            f"max_total_runtime_minutes={max_total_runtime_minutes}\n"
            f"reminder_interval_minutes={interval}\n"
            "session_started_utc=parse_failed"
        )
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    t0 = t0.astimezone(timezone.utc)
    elapsed = (now - t0).total_seconds() / 60.0
    remaining = max(0.0, float(max_total_runtime_minutes) - elapsed)
    deadline = t0 + timedelta(minutes=float(max_total_runtime_minutes))
    next_reminder_elapsed = min(max_total_runtime_minutes, (int(elapsed // float(interval)) + 1) * interval)
    next_reminder_utc = t0 + timedelta(minutes=float(next_reminder_elapsed))
    return (
        f"current_utc={now.isoformat()}\n"
        f"session_started_utc={t0.isoformat()}\n"
        f"deadline_utc={deadline.isoformat()}\n"
        f"max_total_runtime_minutes={max_total_runtime_minutes}\n"
        f"reminder_interval_minutes={interval}\n"
        f"next_reminder_utc={next_reminder_utc.isoformat()}\n"
        f"elapsed_minutes≈{elapsed:.2f}\n"
        f"remaining_minutes≈{remaining:.2f}"
    )


def programmer_runtime_reminder(
    *,
    session_started_utc_iso: str | None,
    max_total_runtime_minutes: int,
    reminder_interval_minutes: int = 5,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
        reminder_interval_minutes=reminder_interval_minutes,
    )
    return f"""运行中时间提醒。总预算从 Python 主程序启动计时为 {max_total_runtime_minutes} 分钟。
剩余时间不足 3 分钟时，不要再启动新优化方向，应确保已有 correctness-passed 候选被 promote 并交给汇总。

{tb}"""


def planner_user_message(
    targets: list[str],
    *,
    session_started_utc_iso: str | None = None,
    max_total_runtime_minutes: int = 25,
    reminder_interval_minutes: int = 5,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
        reminder_interval_minutes=reminder_interval_minutes,
    )
    return f"""<targets>
{targets!r}
</targets>

<time_budget>
{tb}
</time_budget>

请根据 Phase3 约束输出计划。"""


def supervisor_user_message(
    *,
    targets: list[str],
    plan: str,
    evidence_tail: str,
    programmer_rounds: int,
    max_rounds: int,
    session_started_utc_iso: str | None = None,
    max_total_runtime_minutes: int = 25,
    reminder_interval_minutes: int = 5,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
        reminder_interval_minutes=reminder_interval_minutes,
    )
    return f"""
<targets>{targets!r}</targets>
<time_budget>
{tb}
</time_budget>
<plan_excerpt>
{plan[:4000]}
</plan_excerpt>
<programmer_rounds>{programmer_rounds}</programmer_rounds>
<max_programmer_rounds>{max_rounds}</max_programmer_rounds>
<evidence_tail>
{evidence_tail[:14000]}
</evidence_tail>
请严格遵守 <routing_rules>，只输出 JSON。
""".strip()


def programmer_user_message(
    *,
    targets: list[str],
    plan: str,
    evidence_so_far: str,
    round_index: int,
    max_rounds: int,
    session_started_utc_iso: str | None = None,
    max_total_runtime_minutes: int = 25,
    reminder_interval_minutes: int = 5,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
        reminder_interval_minutes=reminder_interval_minutes,
    )
    return f"""
<targets>
{targets!r}
</targets>

<time_budget>
{tb}
</time_budget>

<plan>
{plan}
</plan>

<prior_evidence>
{evidence_so_far[:10000] if evidence_so_far else "(首轮尚无)"}
</prior_evidence>

<round>
当前为编程-执行子智能体第 {round_index} / {max_rounds} 轮。
</round>

请遵循系统消息中的工具协议开始工作。首步应读取 Phase3 上下文；若已有失败证据，先修复 correctness。
""".strip()


def synthesizer_user_message(
    *,
    targets: list[str],
    plan: str,
    evidence: str,
    session_started_utc_iso: str | None = None,
    max_total_runtime_minutes: int = 25,
    reminder_interval_minutes: int = 5,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
        reminder_interval_minutes=reminder_interval_minutes,
    )
    return f"""
<targets>{targets!r}</targets>
<time_budget>
{tb}
</time_budget>
<plan_excerpt>{plan[:6000]}</plan_excerpt>
<evidence>
{evidence[:60000]}
</evidence>
请根据 <constraints> 只输出 JSON 对象。
""".strip()
