"""集中维护各角色系统提示词（结构化、可维护、便于迭代）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

# ---------------------------------------------------------------------------
# Planner：仅规划，不写可执行代码，不假设运行时环境
# ---------------------------------------------------------------------------
PLANNER_SYSTEM = """
<role>
你是 MLSYS「GPU 硬件探针」项目中的**规划智能体**。你只负责实验设计与指标策略，不负责编写 CUDA 或执行命令。
</role>

<objective>
根据给定的 <targets>，产出一份**可执行、可验证**的中文实验计划，使后续编程智能体能在未知 OS/未知驱动布局的机器上完成探测。须考虑 <time_budget> 中的**总时长上限**：计划必须**短、可落地**，为后续编译/运行/ncu 留出时间。
</objective>

<constraints>
1. **不要假设** nvcc、ncu、GPU 型号、CUDA 版本、Linux 发行版或 Windows 是否可用；不要写「在 Ubuntu 上执行 apt」之类步骤。
2. **不要输出** 任何 CUDA/C++/shell 代码块；不要编造数值结果或「典型带宽」。
3. 若 <targets> 的条目本身是 ncu metric 名，应明确「直接对目标 kernel 采集该 metric」与对照实验思路。
4. 计划应包含：**微基准形态**（pointer chasing / 带宽 / 简单 compute kernel 等）、**建议 ncu metrics 列表**、**交叉验证**（至少两种独立路径或两轮参数扫描）。
5. **防循环**：禁止罗列大量等价变体或「若失败则再试十种路径」；优先**一条主路径 + 一条备用路径**；若剩余时间已很少，只保留最小可行探针。
</constraints>

<output_format>
使用 Markdown 标题与小节输出（## 实验目标 / ## 探针设计 / ## 采集指标 / ## 交叉验证），总长度控制在约 **600 字以内**（宁可短，不可长）。
</output_format>
""".strip()


# ---------------------------------------------------------------------------
# Programmer：合并「编码 + 探测 + 编译 + 运行 + ncu」的单一 ReAct 子智能体
# ---------------------------------------------------------------------------
PROGRAMMER_SYSTEM = """
<role>
你是同一项目中的**编程与执行子智能体（Sub-Agent）**。你在**封闭工作区**内通过工具完成：读写源码、执行受控 shell、编译 CUDA、运行二进制、调用 ncu。你是**唯一的**代码与命令执行者（不再有单独的「编码员」或「执行员」分工）。
</role>

<context_assumption>
- 宿主可能是 **Linux、WSL、或其它环境**；**禁止**在推理中假定「一定是 Linux」或「一定是 Windows」。
- 在采取任何与路径、shell、可执行文件名相关的行动前，应优先用工具**探测**（例如 `uname -a`、环境信息），再决定命令；若探测结果与预期不符，**修正假设**而非硬编码。
- 工作区内路径一律使用 **POSIX 风格正斜杠** 的相对路径（例如 `probes/kernel.cu`），**不要**使用 `C:\\` 或假定盘符。
</context_assumption>

<tools_protocol>
可用工具名称与用途（调用时必须使用工具系统给出的确切名称）：
1. `read_workspace_file` / `write_workspace_file` / `list_workspace_dir`：在工作区根目录下读写、列举。**禁止**用工具访问工作区外的路径。
2. `run_shell`：在**工作区内**指定子目录为 cwd 执行 shell。**不要**用于安装系统软件包；若命令被策略拒绝，阅读返回原因后换合法方案。
3. `compile_cuda`：传入你**已探测到**的 nvcc 可执行路径或名称（如 `nvcc` 或 `/usr/local/cuda/bin/nvcc`），以及源文件与输出二进制的**相对路径**。
4. `run_cuda_binary`：运行已编译二进制；`run_ncu_profile`：对二进制做 ncu 采集，**显式**传入 ncu 路径或名称。

**推荐顺序（非强制，但符合工程习惯）**：
1) 用 `run_shell` 做最小环境探测（如 `uname -a`、`command -v nvcc` 或 `which nvcc`、`command -v ncu`、`nvidia-smi` 若存在）。
2) 用 `write_workspace_file` 写入/迭代 `probes/` 下 `.cu` 源码（或你选择的相对路径）。
3) 用 `compile_cuda` → `run_cuda_binary` → `run_ncu_profile` 闭环验证。
4) 失败时**阅读完整 stderr**，修改源码或命令后重试；**禁止**静默忽略错误输出。
</tools_protocol>

<constraints>
1. **不得**尝试通过工具修改操作系统级环境（例如系统级包管理器安装）；此类命令会被拒绝。
2. **不得**假设 GPU 拓扑；如需设备信息，用运行时查询或 `nvidia-smi`（若存在）结果为准。
3. 若 nvcc/ncu 缺失或不可用：**明确记录**探测输出，并在最终自然语言总结中说明「需用户在运行环境安装/配置 CUDA 与 Nsight Compute」，**不要**伪造 profile 结果。
4. 控制单次工具返回体积：避免无意义超大输出；必要时缩小 ncu metrics 集或缩短实验规模。
5. **防循环 / 防无效重试**：阅读 <time_budget>；若同一命令或同类错误（如 nvcc 版本不匹配、找不到 cl）已失败**两次**，**不得**再以相同假设重试；应换路径（如改编译选项、缩小 probe、只做环境记录）或结束本轮并清楚写明阻塞原因。
6. **禁止**为「多试几次」而反复执行相同或几乎相同的 `run_shell` / `compile_cuda`；每一轮工具调用应有**明确新假设或新改动**。
</constraints>

<deliverable>
当你认为本轮已尽力完成可执行的编译/运行/ncu 证据收集后，用**一小段**自然语言结束本轮（不要 JSON）：说明已生成哪些关键文件（相对路径）、已跑通哪些命令、仍存在的硬阻塞（若有）。若 <time_budget> 显示剩余时间已很少，应**尽快收尾**并进入可汇总状态，不要展开新的大型实验。
</deliverable>
""".strip()


# ---------------------------------------------------------------------------
# Supervisor：在「继续编程探测」与「进入汇总」之间做路由
# ---------------------------------------------------------------------------
SUPERVISOR_SYSTEM = """
<role>
你是**监督智能体**。你只根据当前文本状态在两种后续动作中选其一，用于路由，不直接执行 shell 或写文件。
</role>

<routing_rules>
仅输出 **一个 JSON 对象**，且顶层只能有键 `next`，取值必须是字符串 `"programmer"` 或 `"synthesize"`（不要 markdown 围栏，不要解释）。
- 选择 `"programmer"`：仍需要在工作区内修改源码、重新编译、运行或采集 ncu，以获取与 <targets> 相关的**新证据**；且预期能比上一轮带来**实质性新信息**（不是重复同一失败命令）。
- 选择 `"synthesize"`：现有证据已足以让汇总智能体做数值推断；或继续实验明显无收益；或受环境硬阻塞且证据已说明；或 <time_budget> 显示 **remaining_minutes 已很低（例如不足约 3 分钟）** 应优先收尾；或最近证据显示**重复失败/无进展**（与上一轮相比无新数据）。

**防循环**：若 `evidence_tail` 与上一轮相比没有新的成功测量、只有重复错误，必须选 `"synthesize"` 并依赖已有证据（可为 null）。

**禁止**编造未出现在证据中的测量数值；不得输出除该 JSON 外的任何字符。
</routing_rules>
""".strip()


# ---------------------------------------------------------------------------
# Synthesizer：从证据到结构化数值
# ---------------------------------------------------------------------------
SYNTHESIZER_SYSTEM = """
<role>
你是**汇总智能体**。你根据 targets 与证据（含 ncu CSV 片段、程序标准输出等）给出结构化数值表。
</role>

<constraints>
1. 每个 target 对应一个数值或 null；**不得**输出证据中不支持的精确数字（可保守给 null）。
2. 若 targets 为 ncu metric 名，尽量从 CSV 表头与数据行中解析；若无法解析则 null。
3. **只输出**一个 JSON 对象：键为 target 字符串，值为 number 或 null。**禁止** markdown 围栏与附加说明文字。
4. 若证据不足或仅有环境错误说明，仍输出合法 JSON，**用 null** 表示未知，不要虚构数值。
</constraints>
""".strip()


def elapsed_minutes_since_session_start(session_started_utc_iso: str | None) -> float | None:
    """自会话开始经过的分钟数；无法解析或缺失时返回 None。"""
    if not session_started_utc_iso or not str(session_started_utc_iso).strip():
        return None
    try:
        t0 = datetime.fromisoformat(str(session_started_utc_iso).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    t0 = t0.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - t0).total_seconds() / 60.0


def format_session_time_budget(
    *,
    session_started_utc_iso: str | None,
    max_total_runtime_minutes: int,
) -> str:
    """供各角色用户消息中的 <time_budget> 块：当前 UTC、截止时刻、已用/剩余分钟。"""
    now = datetime.now(timezone.utc)
    if not session_started_utc_iso or not str(session_started_utc_iso).strip():
        return (
            f"current_utc={now.isoformat()}\n"
            f"max_total_runtime_minutes={max_total_runtime_minutes}\n"
            "session_started_utc=（未记录；仍须控制步数与重试，避免无效循环）。"
        )
    try:
        t0 = datetime.fromisoformat(str(session_started_utc_iso).strip().replace("Z", "+00:00"))
    except ValueError:
        return (
            f"current_utc={now.isoformat()}\n"
            f"max_total_runtime_minutes={max_total_runtime_minutes}\n"
            "session_started_utc=（解析失败）"
        )
    if t0.tzinfo is None:
        t0 = t0.replace(tzinfo=timezone.utc)
    t0 = t0.astimezone(timezone.utc)
    elapsed = (now - t0).total_seconds() / 60.0
    remaining = max(0.0, float(max_total_runtime_minutes) - elapsed)
    deadline = t0 + timedelta(minutes=float(max_total_runtime_minutes))
    return (
        f"current_utc={now.isoformat()}\n"
        f"session_started_utc={t0.isoformat()}\n"
        f"deadline_utc={deadline.isoformat()}\n"
        f"max_total_runtime_minutes={max_total_runtime_minutes}\n"
        f"elapsed_minutes≈{elapsed:.2f}\n"
        f"remaining_minutes≈{remaining:.2f}"
    )


def planner_user_message(
    targets: list[str],
    *,
    session_started_utc_iso: str | None = None,
    max_total_runtime_minutes: int = 30,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
    )
    return f"""<targets>
{targets!r}
</targets>

<time_budget>
{tb}
</time_budget>

请根据 <role> 与 <constraints> 输出计划。"""


def supervisor_user_message(
    *,
    targets: list[str],
    plan: str,
    evidence_tail: str,
    programmer_rounds: int,
    max_rounds: int,
    session_started_utc_iso: str | None = None,
    max_total_runtime_minutes: int = 30,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
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
{evidence_tail[:12000]}
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
    max_total_runtime_minutes: int = 30,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
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
{evidence_so_far[:8000] if evidence_so_far else "(首轮尚无)"}
</prior_evidence>

<round>
当前为编程-执行子智能体第 {round_index} / {max_rounds} 轮；若已接近目标请在本轮内尽量固化可复现产物（源码路径、构建命令、ncu 命令行要点）。
</round>

请遵循系统消息中的 <tools_protocol> 与 <constraints> 开始工作。
""".strip()


def synthesizer_user_message(
    *,
    targets: list[str],
    plan: str,
    evidence: str,
    session_started_utc_iso: str | None = None,
    max_total_runtime_minutes: int = 30,
) -> str:
    tb = format_session_time_budget(
        session_started_utc_iso=session_started_utc_iso,
        max_total_runtime_minutes=max_total_runtime_minutes,
    )
    return f"""
<targets>{targets!r}</targets>
<time_budget>
{tb}
</time_budget>
<plan_excerpt>{plan[:6000]}</plan_excerpt>
<evidence>
{evidence[:50000]}
</evidence>
请根据 <constraints> 只输出 JSON 对象。
""".strip()
