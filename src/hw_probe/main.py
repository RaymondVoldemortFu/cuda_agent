from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
import uuid
from pathlib import Path

from hw_probe.agents.graph import run_probe_graph
from hw_probe.config.settings import bootstrap_settings
from hw_probe.observability.llm_trace import JsonlLlmTraceHandler
from hw_probe.runtime.shutdown import install_shutdown_handlers
from hw_probe.observability.logging_setup import configure_logging, parse_console_level
from hw_probe.observability.status_report import print_system_status
from hw_probe.services.output_writer import append_results_log, write_output_json
from hw_probe.services.target_spec import load_target_spec


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _seed_default_probe(workspace: Path) -> None:
    """若工作区尚无 probes/kernel.cu，则拷贝内置模板，便于首轮编译冒烟。"""
    rel = Path("probes/kernel.cu")
    dst = (workspace / rel).resolve()
    try:
        dst.relative_to(workspace.resolve())
    except ValueError:
        return
    if dst.is_file():
        return
    tpl = _package_dir() / "templates" / "gemm_stub.cu"
    if not tpl.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tpl, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLSYS GPU 硬件探针 Agent（LangGraph + LangChain）")
    parser.add_argument(
        "--dev",
        action="store_true",
        help="开发模式：加载 .env（不覆盖已注入环境变量）并默认 development",
    )
    parser.add_argument(
        "--console-log-level",
        type=str,
        default=None,
        metavar="LEVEL",
        help="覆盖控制台日志级别：DEBUG, INFO, WARNING, ERROR（默认取配置或 INFO）",
    )
    args = parser.parse_args()
    install_shutdown_handlers()

    settings = bootstrap_settings(dev=args.dev)
    try:
        settings.validate_llm_config()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    console_override = parse_console_level(args.console_log_level) if args.console_log_level else None
    configure_logging(settings, console_level_override=console_override)
    print_system_status(settings)

    ws = settings.resolved_workspace()
    ws.mkdir(parents=True, exist_ok=True)

    log_dir = settings.resolved_log_dir()
    trace_path = log_dir / settings.llm_trace_jsonl_filename
    session_id = str(uuid.uuid4())
    trace_handler = JsonlLlmTraceHandler(trace_path, session_id=session_id)
    trace_handler.emit_custom(
        "session_start",
        {
            "argv": sys.argv,
            "workspace": str(ws),
            "target_spec": str(settings.target_spec_path.expanduser().resolve()),
        },
    )

    try:
        targets = load_target_spec(settings.target_spec_path.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"读取 target 规格失败: {exc}", file=sys.stderr)
        sys.exit(3)

    append_results_log(ws, settings.results_log_name, f"启动: targets={targets!r} session={session_id}")

    _seed_default_probe(ws)

    try:
        final = run_probe_graph(settings, targets=targets, trace_callbacks=[trace_handler])
    except KeyboardInterrupt:
        append_results_log(ws, settings.results_log_name, "会话被用户中断 (KeyboardInterrupt)")
        trace_handler.emit_custom("session_interrupted", {"reason": "KeyboardInterrupt"})
        print("\n[hw_probe] 已收到终止信号，正在退出。", file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except Exception:
        append_results_log(ws, settings.results_log_name, traceback.format_exc())
        trace_handler.emit_custom(
            "session_error",
            {"traceback": traceback.format_exc()},
        )
        raise
    else:
        trace_handler.emit_custom(
            "session_end",
            {"results_keys": list((final.get("results") or {}).keys())},
        )

    results = final.get("results") or {}
    methodology = str(final.get("methodology") or "")
    evidence = list(final.get("evidence_log") or [])

    out_path = write_output_json(
        ws,
        settings.output_filename,
        results=results,
        methodology=methodology,
        evidence=evidence[-20:],
    )
    append_results_log(ws, settings.results_log_name, f"完成: 已写入 {out_path} session={session_id}")
    print(f"[hw_probe] 输出: {out_path}", flush=True)
    print(f"[hw_probe] 完整 LLM trace: {trace_path}", flush=True)
    print(f"[hw_probe] DEBUG 日志: {log_dir / settings.debug_log_filename}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[hw_probe] 已中断。", file=sys.stderr, flush=True)
        raise SystemExit(130) from None
