from __future__ import annotations

import json
import math
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Annotated, Any

import torch
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from torch.utils.cpp_extension import load

from hw_probe.config.settings import AppSettings
from hw_probe.observability.logging_setup import get_hw_probe_logger
from hw_probe.tools.pathutil import assert_under_workspace

_LOG = get_hw_probe_logger("tools.lora")

_RANK = 16
_DEFAULT_SHAPES = (3584, 3601, 4096)


def baseline_lora_source() -> str:
    """A conservative, compilable seed. The agent is expected to improve candidates."""
    return r'''
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT32(x) TORCH_CHECK((x).scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_DIM2(x) TORCH_CHECK((x).dim() == 2, #x " must be rank-2")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_FLOAT32(x);    \
  CHECK_DIM2(x)

namespace {

void check_inputs(const torch::Tensor& W,
                  const torch::Tensor& X,
                  const torch::Tensor& A,
                  const torch::Tensor& B) {
  CHECK_INPUT(W);
  CHECK_INPUT(X);
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  const auto d = W.size(0);
  TORCH_CHECK(W.size(1) == d, "W must be [d, d]");
  TORCH_CHECK(X.size(0) == d && X.size(1) == d, "X must be [d, d]");
  TORCH_CHECK(A.size(0) == d && A.size(1) == 16, "A must be [d, 16]");
  TORCH_CHECK(B.size(0) == d && B.size(1) == 16, "B must be [d, 16]");
}

}  // namespace

torch::Tensor forward(torch::Tensor W,
                      torch::Tensor X,
                      torch::Tensor A,
                      torch::Tensor B) {
  check_inputs(W, X, A, B);
  auto Y = torch::matmul(W, X);
  auto T = torch::matmul(B.transpose(0, 1).contiguous(), X);
  Y.add_(torch::matmul(A, T));
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "LoRA forward baseline");
}
'''.lstrip()


def seed_initial_optimized_lora(workspace: Path) -> Path:
    path = workspace / "optimized_lora.cu"
    if not path.is_file():
        path.write_text(baseline_lora_source(), encoding="utf-8")
    return path


class EvaluateLoraCandidateArgs(BaseModel):
    source_relative: Annotated[str, Field(description="Candidate .cu path relative to workspace, e.g. candidates/r1.cu")]
    candidate_name: Annotated[str, Field(description="Short stable name for this candidate")]
    shapes: Annotated[
        str,
        Field(default="", description="Comma-separated d values in [3584,4608]. Empty uses 3584,4096."),
    ] = ""
    warmup: Annotated[int, Field(default=3, ge=1, le=20, description="CUDA-event benchmark warmup count")] = 3
    iters: Annotated[int, Field(default=8, ge=1, le=50, description="CUDA-event benchmark iterations")] = 8
    promote_if_best: Annotated[
        bool,
        Field(default=True, description="Copy the candidate to optimized_lora.cu if it is correct and beats current best"),
    ] = True


def _parse_shapes(raw: str) -> list[int]:
    if not raw.strip():
        return list(_DEFAULT_SHAPES)
    out: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        d = int(text)
        if d < 3584 or d > 4608:
            raise ValueError(f"d={d} is outside [3584, 4608]")
        out.append(d)
    if not out:
        raise ValueError("no valid shapes provided")
    if all(d % 4 == 0 for d in out):
        out.append(3601)
    return out


def _safe_module_name(candidate_name: str) -> str:
    clean = re.sub(r"[^0-9a-zA-Z_]+", "_", candidate_name).strip("_") or "candidate"
    return f"optimized_lora_ext_{clean}_{int(time.time() * 1000)}"


def _make_inputs(d: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(20260512 + d)
    scale = 1.0 / math.sqrt(float(d))
    device = torch.device("cuda")
    W = (torch.randn((d, d), generator=gen, dtype=torch.float32) * scale).contiguous().to(device)
    X = (torch.randn((d, d), generator=gen, dtype=torch.float32) * scale).contiguous().to(device)
    A = (torch.randn((d, _RANK), generator=gen, dtype=torch.float32) * scale).contiguous().to(device)
    B = (torch.randn((d, _RANK), generator=gen, dtype=torch.float32) * scale).contiguous().to(device)
    return W, X, A, B


def _reference(W: torch.Tensor, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return W @ X + A @ (B.transpose(0, 1).contiguous() @ X)


def _benchmark(fn: Any, W: torch.Tensor, X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, warmup: int, iters: int) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = fn(W, X, A, B)
        torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn(W, X, A, B)
            end.record()
            torch.cuda.synchronize()
            times.append(float(start.elapsed_time(end)))
    times.sort()
    return times[len(times) // 2]


def _read_best(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"score": 0.0, "candidate_name": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def make_lora_tools(settings: AppSettings) -> list[StructuredTool]:
    ws = settings.resolved_workspace()
    max_chars = settings.max_tool_output_chars

    def evaluate_lora_candidate(
        source_relative: str,
        candidate_name: str,
        shapes: str = "",
        warmup: int = 3,
        iters: int = 8,
        promote_if_best: bool = True,
    ) -> str:
        _LOG.debug("evaluate_lora_candidate source=%r candidate=%r shapes=%r", source_relative, candidate_name, shapes)
        source = assert_under_workspace(ws, source_relative)
        if not source.is_file():
            raise FileNotFoundError(f"candidate source does not exist: {source_relative}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to evaluate LoRA candidates")

        selected_shapes = _parse_shapes(shapes)
        payload: dict[str, Any] = {
            "candidate_name": candidate_name,
            "source_relative": source_relative,
            "device": torch.cuda.get_device_name(0),
            "compile_ok": False,
            "correct": False,
            "promoted": False,
            "score": 0.0,
            "shapes": [],
        }
        try:
            module = load(
                name=_safe_module_name(candidate_name),
                sources=[str(source)],
                verbose=False,
                extra_cuda_cflags=["-O3"],
                with_cuda=True,
            )
            payload["compile_ok"] = True
            speedups: list[float] = []
            for d in selected_shapes:
                W, X, A, B = _make_inputs(d)
                with torch.no_grad():
                    y_student = module.forward(W, X, A, B)
                    y_ref = _reference(W, X, A, B)
                    diff = (y_student - y_ref).float()
                    max_abs_err = float(diff.abs().max().item())
                    rel_l2_err = float((diff.norm() / (y_ref.float().norm() + 1e-12)).item())
                    correct = bool(torch.allclose(y_student, y_ref, rtol=1e-4, atol=1e-4))
                torch_ms = _benchmark(_reference, W, X, A, B, warmup, iters)
                student_ms: float | None = None
                speedup = 0.0
                if correct:
                    student_ms = _benchmark(module.forward, W, X, A, B, warmup, iters)
                    speedup = torch_ms / student_ms
                    speedups.append(speedup)
                payload["shapes"].append(
                    {
                        "d": d,
                        "correct": correct,
                        "max_abs_err": max_abs_err,
                        "rel_l2_err": rel_l2_err,
                        "torch_median_ms": torch_ms,
                        "student_median_ms": student_ms,
                        "speedup": speedup,
                    }
                )
                if not correct:
                    break
            payload["correct"] = bool(payload["shapes"]) and all(s["correct"] for s in payload["shapes"])
            if payload["correct"] and speedups:
                payload["score"] = sum(speedups) / len(speedups)
                best_path = ws / "stage2_best.json"
                best = _read_best(best_path)
                candidate_shapes = {int(s["d"]) for s in payload["shapes"]}
                best_shapes = {int(s["d"]) for s in best.get("shapes") or [] if "d" in s}
                better_score = float(payload["score"]) > float(best.get("score") or 0.0)
                broader_safe_coverage = bool(candidate_shapes - best_shapes) and float(payload["score"]) >= (
                    0.98 * float(best.get("score") or 0.0)
                )
                if promote_if_best and (better_score or broader_safe_coverage):
                    tmp = ws / "optimized_lora.cu.next"
                    shutil.copyfile(source, tmp)
                    tmp.replace(ws / "optimized_lora.cu")
                    payload["promoted"] = True
                    _write_json(
                        best_path,
                        {
                            "score": payload["score"],
                            "candidate_name": candidate_name,
                            "source_relative": source_relative,
                            "shapes": payload["shapes"],
                        },
                    )
        except Exception:
            payload["error"] = traceback.format_exc()

        out_path = ws / "stage2_candidates" / f"{_safe_module_name(candidate_name)}.json"
        _write_json(out_path, payload)
        payload["result_path"] = str(out_path.relative_to(ws))
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if len(text) > max_chars:
            half = max_chars // 2
            text = text[:half] + "\n...[truncated]...\n" + text[-half:]
        return text

    return [
        StructuredTool.from_function(
            name="evaluate_lora_candidate",
            description=(
                "Compile a candidate optimized_lora.cu with torch.utils.cpp_extension.load, "
                "test correctness for synthetic FP32 LoRA inputs, benchmark with CUDA events, "
                "and atomically promote it to workspace/optimized_lora.cu only if it is correct and faster than current best."
            ),
            args_schema=EvaluateLoraCandidateArgs,
            func=evaluate_lora_candidate,
        )
    ]
