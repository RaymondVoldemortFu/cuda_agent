from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from hw_probe.config.settings import AppSettings
from hw_probe.observability.logging_setup import get_hw_probe_logger
from hw_probe.runtime.shutdown import interruptible_communicate
from hw_probe.tools.pathutil import assert_under_workspace

_LOG = get_hw_probe_logger("tools.phase3")


class ReadPhase3ContextArgs(BaseModel):
    include_evaluator_excerpt: Annotated[
        bool,
        Field(default=True, description="Whether to include excerpts from public evaluator files."),
    ] = True


class WriteEngineCandidateArgs(BaseModel):
    candidate_name: Annotated[str, Field(description="Short stable candidate name, e.g. kv_cache_v1")]
    content: Annotated[str, Field(description="Complete Python source for a candidate engine.py")]


class InspectEngineArgs(BaseModel):
    relative_path: Annotated[
        str,
        Field(description="Workspace-relative path, e.g. phase3_candidates/kv_cache_v1.py or engine.py"),
    ]
    max_chars: Annotated[int, Field(default=40000, ge=1000, le=200000)] = 40000


class RunCandidateArgs(BaseModel):
    candidate_relative: Annotated[
        str,
        Field(description="Workspace-relative candidate path, e.g. phase3_candidates/kv_cache_v1.py"),
    ]
    device: Annotated[str, Field(default="auto", description="Evaluator device argument. Use auto unless debugging.")]


class PromoteCandidateArgs(BaseModel):
    candidate_relative: Annotated[str, Field(description="Workspace-relative candidate path to promote.")]
    candidate_name: Annotated[str, Field(description="Candidate name used in phase3_best.json.")]
    correctness_result_path: Annotated[
        str,
        Field(description="Workspace-relative JSON result written by run_engine_correctness."),
    ]
    benchmark_result_path: Annotated[
        str | None,
        Field(default=None, description="Workspace-relative JSON result written by run_engine_benchmark."),
    ] = None


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name.strip())
    cleaned = cleaned.strip("_-") or f"candidate_{int(time.time())}"
    return cleaned[:80]


def _repo_root_from_workspace(ws: Path) -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "stage3").exists() or (cwd / "run.sh").exists():
        return cwd
    resolved = ws.resolve()
    if resolved.name == "workspace":
        return resolved.parent
    return resolved


def _resolve_existing_path(configured: Path, fallbacks: list[Path]) -> Path:
    candidates = [configured, *fallbacks]
    for p in candidates:
        p = p.expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        if p.exists():
            return p
    rendered = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not resolve an existing Phase3 path. Tried:\n{rendered}")


def _resolve_model_config(settings: AppSettings, ws: Path) -> Path:
    root = _repo_root_from_workspace(ws)
    return _resolve_existing_path(
        settings.phase3_model_config_path,
        [
            root / "target" / "model_config.json",
            root / "stage3" / "target" / "model_config.json",
            Path("/target/model_config.json"),
        ],
    )


def _resolve_weight_dir(settings: AppSettings, ws: Path) -> Path:
    root = _repo_root_from_workspace(ws)
    return _resolve_existing_path(
        settings.phase3_weight_dir,
        [
            root / "target" / "weights",
            root / "stage3" / "target" / "weights",
            Path("/target/weights"),
        ],
    )


def _resolve_weight_dir_for_context(settings: AppSettings, ws: Path) -> Path:
    try:
        return _resolve_weight_dir(settings, ws)
    except FileNotFoundError:
        configured = settings.phase3_weight_dir.expanduser()
        if configured.is_absolute():
            return configured.resolve()
        return (Path.cwd() / configured).resolve()


def _resolve_evaluator_dir(settings: AppSettings, ws: Path) -> Path:
    root = _repo_root_from_workspace(ws)
    configured = settings.phase3_evaluator_dir or Path("stage3/evaluator")
    return _resolve_existing_path(
        configured,
        [
            root / "evaluator",
            root / "stage3" / "evaluator",
            Path("/evaluator"),
        ],
    )


def _run_python(
    args: list[str],
    *,
    cwd: Path,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        [sys.executable, *args],
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = interruptible_communicate(proc, total_timeout_sec=float(timeout_sec))
    rc = proc.returncode
    if rc is None:
        rc = 0
    return subprocess.CompletedProcess([sys.executable, *args], rc, stdout or "", stderr or "")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]


def _result_text(payload: dict[str, Any], max_chars: int) -> str:
    return _truncate(json.dumps(payload, indent=2, ensure_ascii=False), max_chars)


def _benchmark_score(payload: dict[str, Any]) -> float:
    if not payload.get("passed"):
        return 0.0
    parsed = payload.get("parsed_results")
    if not isinstance(parsed, list):
        return 0.0
    by_name = {str(item.get("case_name")): item for item in parsed if isinstance(item, dict)}
    prefill = float((by_name.get("prefill") or {}).get("tokens_per_second") or 0.0)
    decode = float((by_name.get("decode") or {}).get("decode_tokens_per_second") or 0.0)
    mixed = float((by_name.get("mixed") or {}).get("tokens_per_second") or 0.0)
    # Decode dominates serving traces, but keep prefill/mixed visible in promotion.
    return decode * 0.5 + mixed * 0.3 + prefill * 0.2


def make_phase3_tools(settings: AppSettings) -> list[StructuredTool]:
    ws = settings.resolved_workspace()
    max_chars = settings.max_tool_output_chars

    def read_phase3_context(include_evaluator_excerpt: bool = True) -> str:
        model_config_path = _resolve_model_config(settings, ws)
        weight_dir = _resolve_weight_dir_for_context(settings, ws)
        evaluator_dir = _resolve_evaluator_dir(settings, ws)
        config = json.loads(model_config_path.read_text(encoding="utf-8"))
        payload: dict[str, Any] = {
            "workspace": str(ws),
            "model_config_path": str(model_config_path),
            "weight_dir": str(weight_dir),
            "weight_dir_exists": weight_dir.is_dir(),
            "weight_files": sorted(p.name for p in weight_dir.iterdir()) if weight_dir.is_dir() else [],
            "evaluator_dir": str(evaluator_dir),
            "model_config": config,
            "required_output": {
                "engine": str(ws / "engine.py"),
                "results_log": str(ws / settings.results_log_name),
                "output": str(ws / settings.output_filename),
            },
            "engine_contract": [
                "The engine defines create_engine(model_config, weight_dir, device='cuda').",
                "The returned object implements prefill(request_ids, input_ids), decode(request_ids, token_ids), and remove(request_ids).",
                "prefill/decode return last-token logits shaped [batch_size, vocab_size].",
                "Model dimensions and dtype behavior are derived from model_config.",
                "Promotion requires a passing run_engine_correctness result.",
            ],
        }
        if include_evaluator_excerpt:
            excerpts: dict[str, str] = {}
            for name in ("test_correctness.py", "benchmark_throughput.py", "reference_model.py"):
                path = evaluator_dir / name
                if path.is_file():
                    excerpts[name] = _truncate(path.read_text(encoding="utf-8"), 24000)
            payload["evaluator_excerpts"] = excerpts
        return _result_text(payload, max_chars)

    def write_engine_candidate(candidate_name: str, content: str) -> str:
        safe = _safe_name(candidate_name)
        rel = Path(settings.phase3_candidate_dir_name) / f"{safe}.py"
        path = assert_under_workspace(ws, rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        payload = {
            "candidate_name": safe,
            "candidate_relative": str(rel),
            "bytes": len(content.encode("utf-8")),
            "path": str(path),
        }
        return _result_text(payload, max_chars)

    def inspect_engine_file(relative_path: str, max_chars: int = 40000) -> str:
        path = assert_under_workspace(ws, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"engine file does not exist: {relative_path}")
        return _truncate(path.read_text(encoding="utf-8"), max_chars)

    def run_engine_correctness(candidate_relative: str, device: str = "auto") -> str:
        candidate = assert_under_workspace(ws, candidate_relative)
        if not candidate.is_file():
            raise FileNotFoundError(f"candidate does not exist: {candidate_relative}")
        model_config = _resolve_model_config(settings, ws)
        weight_dir = _resolve_weight_dir(settings, ws)
        evaluator_dir = _resolve_evaluator_dir(settings, ws)
        script = evaluator_dir / "test_correctness.py"
        if not script.is_file():
            raise FileNotFoundError(f"missing correctness evaluator: {script}")

        result_dir = ws / settings.phase3_candidate_dir_name / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{candidate.stem}_correctness.json"
        payload: dict[str, Any] = {
            "candidate_relative": str(Path(candidate_relative)),
            "candidate_path": str(candidate),
            "model_config": str(model_config),
            "weight_dir": str(weight_dir),
            "evaluator": str(script),
            "device": device,
            "passed": False,
        }
        try:
            proc = _run_python(
                [
                    str(script),
                    "--engine",
                    str(candidate),
                    "--model-config",
                    str(model_config),
                    "--weight-dir",
                    str(weight_dir),
                    "--device",
                    device,
                ],
                cwd=evaluator_dir,
                timeout_sec=settings.phase3_correctness_timeout_sec,
            )
            payload.update(
                {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "passed": proc.returncode == 0,
                }
            )
            try:
                payload["parsed_stdout"] = json.loads(proc.stdout)
            except json.JSONDecodeError:
                payload["parsed_stdout"] = None
        except Exception:
            payload["error"] = traceback.format_exc()
        _write_json(result_path, payload)
        payload["result_relative"] = str(result_path.relative_to(ws))
        return _result_text(payload, max_chars)

    def run_engine_benchmark(candidate_relative: str, device: str = "auto") -> str:
        candidate = assert_under_workspace(ws, candidate_relative)
        if not candidate.is_file():
            raise FileNotFoundError(f"candidate does not exist: {candidate_relative}")
        model_config = _resolve_model_config(settings, ws)
        weight_dir = _resolve_weight_dir(settings, ws)
        evaluator_dir = _resolve_evaluator_dir(settings, ws)
        script = evaluator_dir / "benchmark_throughput.py"
        if not script.is_file():
            raise FileNotFoundError(f"missing benchmark evaluator: {script}")

        result_dir = ws / settings.phase3_candidate_dir_name / "results"
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{candidate.stem}_benchmark.json"
        payload: dict[str, Any] = {
            "candidate_relative": str(Path(candidate_relative)),
            "candidate_path": str(candidate),
            "model_config": str(model_config),
            "weight_dir": str(weight_dir),
            "evaluator": str(script),
            "device": device,
            "passed": False,
            "score": 0.0,
        }
        try:
            proc = _run_python(
                [
                    str(script),
                    "--engine",
                    str(candidate),
                    "--model-config",
                    str(model_config),
                    "--weight-dir",
                    str(weight_dir),
                    "--device",
                    device,
                ],
                cwd=evaluator_dir,
                timeout_sec=settings.phase3_benchmark_timeout_sec,
            )
            payload.update(
                {
                    "exit_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "passed": proc.returncode == 0,
                }
            )
            try:
                parsed = json.loads(proc.stdout)
            except json.JSONDecodeError:
                parsed = None
            payload["parsed_results"] = parsed
            payload["score"] = _benchmark_score(payload)
        except Exception:
            payload["error"] = traceback.format_exc()
        _write_json(result_path, payload)
        payload["result_relative"] = str(result_path.relative_to(ws))
        return _result_text(payload, max_chars)

    def promote_engine_candidate(
        candidate_relative: str,
        candidate_name: str,
        correctness_result_path: str,
        benchmark_result_path: str | None = None,
    ) -> str:
        candidate = assert_under_workspace(ws, candidate_relative)
        correctness_path = assert_under_workspace(ws, correctness_result_path)
        if not candidate.is_file():
            raise FileNotFoundError(f"candidate does not exist: {candidate_relative}")
        correctness = _read_json_if_exists(correctness_path)
        if not correctness.get("passed"):
            raise ValueError("Refusing to promote: correctness_result_path is missing or did not pass.")

        benchmark: dict[str, Any] = {}
        if benchmark_result_path and str(benchmark_result_path).strip():
            benchmark = _read_json_if_exists(assert_under_workspace(ws, benchmark_result_path))

        score = _benchmark_score(benchmark) if benchmark else 0.0
        best_path = ws / "phase3_best.json"
        best = _read_json_if_exists(best_path)
        best_score = float(best.get("score") or 0.0)
        has_best_engine = (ws / "engine.py").is_file()
        should_promote = (not has_best_engine) or score >= best_score or not benchmark

        payload: dict[str, Any] = {
            "candidate_name": candidate_name,
            "candidate_relative": candidate_relative,
            "correctness_result_path": correctness_result_path,
            "benchmark_result_path": benchmark_result_path,
            "score": score,
            "previous_best_score": best_score,
            "promoted": False,
            "reason": "",
        }
        if should_promote:
            tmp = ws / "engine.py.next"
            shutil.copyfile(candidate, tmp)
            tmp.replace(ws / "engine.py")
            best_payload = {
                "candidate_name": candidate_name,
                "candidate_relative": candidate_relative,
                "correctness_result_path": correctness_result_path,
                "benchmark_result_path": benchmark_result_path,
                "score": score,
                "promoted_at_unix": time.time(),
            }
            _write_json(best_path, best_payload)
            payload["promoted"] = True
            payload["reason"] = "correctness passed and candidate is current best or no previous best exists"
        else:
            payload["reason"] = "correctness passed but benchmark score did not improve current best"

        return _result_text(payload, max_chars)

    return [
        StructuredTool.from_function(
            name="read_phase3_context",
            description="Read Phase3 model config, target paths, and public evaluator excerpts for planning an LLM runtime engine.",
            args_schema=ReadPhase3ContextArgs,
            func=read_phase3_context,
        ),
        StructuredTool.from_function(
            name="write_engine_candidate",
            description="Write a complete candidate engine.py source file into the Phase3 candidate directory.",
            args_schema=WriteEngineCandidateArgs,
            func=write_engine_candidate,
        ),
        StructuredTool.from_function(
            name="inspect_engine_file",
            description="Read an existing engine candidate or promoted workspace/engine.py for debugging.",
            args_schema=InspectEngineArgs,
            func=inspect_engine_file,
        ),
        StructuredTool.from_function(
            name="run_engine_correctness",
            description="Run the Phase3 correctness evaluator against a candidate engine. Promotion is forbidden unless this passes.",
            args_schema=RunCandidateArgs,
            func=run_engine_correctness,
        ),
        StructuredTool.from_function(
            name="run_engine_benchmark",
            description="Run the Phase3 throughput benchmark against a correctness-passed candidate engine.",
            args_schema=RunCandidateArgs,
            func=run_engine_benchmark,
        ),
        StructuredTool.from_function(
            name="promote_engine_candidate",
            description="Promote a candidate to workspace/engine.py only when a real correctness result has passed.",
            args_schema=PromoteCandidateArgs,
            func=promote_engine_candidate,
        ),
    ]
