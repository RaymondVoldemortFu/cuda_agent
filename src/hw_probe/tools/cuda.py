from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from hw_probe.config.settings import AppSettings
from hw_probe.observability.logging_setup import get_hw_probe_logger
from hw_probe.runtime.shutdown import interruptible_communicate
from hw_probe.tools.pathutil import assert_under_workspace

_LOG = get_hw_probe_logger("tools.cuda")


def _run_cmd(
    cmd: list[str],
    cwd: Path | None,
    *,
    timeout_sec: float | None,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = interruptible_communicate(proc, total_timeout_sec=timeout_sec)
    rc = proc.returncode
    if rc is None:
        rc = 0
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout or "", stderr=stderr or "")


def compile_cuda_source(
    *,
    nvcc: str,
    source: Path,
    output_binary: Path,
    cwd: Path | None = None,
    timeout_sec: float | None = None,
) -> None:
    output_binary.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nvcc, str(source), "-O3", "-std=c++17", "-o", str(output_binary)]
    result = _run_cmd(cmd, cwd, timeout_sec=timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(
            f"nvcc 编译失败 (exit={result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def run_cuda_binary(
    binary: Path,
    program_args: list[str],
    cwd: Path | None = None,
    *,
    timeout_sec: float | None = None,
) -> str:
    cmd = [str(binary), *program_args]
    result = _run_cmd(cmd, cwd, timeout_sec=timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(
            f"程序运行失败 (exit={result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


def profile_with_ncu(
    *,
    ncu: str,
    binary: Path,
    program_args: list[str],
    metrics: str | None,
    cwd: Path | None = None,
    timeout_sec: float | None = None,
) -> str:
    if metrics is None:
        metrics = ",".join(
            [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__pipe_tensor_op_hmma_cycle_active.avg.pct_of_peak_sustained_active",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "l2__throughput.avg.pct_of_peak_sustained_elapsed",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "sm__maximum_warps_per_active_cycle_pct",
            ]
        )
    cmd = [
        ncu,
        "-f",
        "--target-processes",
        "all",
        "--metrics",
        metrics,
        "--csv",
        str(binary),
        *program_args,
    ]
    result = _run_cmd(cmd, cwd, timeout_sec=timeout_sec)
    if result.returncode != 0:
        raise RuntimeError(
            f"ncu 执行失败 (exit={result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return "=== NCU CSV REPORT ===\n" + result.stdout


class CompileCudaArgs(BaseModel):
    nvcc_executable: Annotated[str, Field(description="nvcc 可执行文件路径或名称（如 nvcc 或 /usr/local/cuda/bin/nvcc）")]
    source_relative: Annotated[str, Field(description=".cu 源文件相对工作区路径")]
    output_binary_relative: Annotated[str, Field(description="输出二进制相对工作区路径")]


class RunBinaryArgs(BaseModel):
    binary_relative: Annotated[str, Field(description="可执行文件相对工作区路径")]
    program_args: Annotated[str, Field(default="", description="传给程序的参数，空格分隔")]


class RunNcuArgs(BaseModel):
    ncu_executable: Annotated[str, Field(description="ncu 可执行文件路径或名称")]
    binary_relative: Annotated[str, Field(description="被 profile 的二进制相对工作区路径")]
    program_args: Annotated[str, Field(default="", description="传给程序的参数，空格分隔")]
    metrics: Annotated[
        str | None,
        Field(default=None, description="逗号分隔的 ncu metrics；为空则使用默认的一组 roofline 相关指标"),
    ] = None


def make_cuda_tools(settings: AppSettings) -> list[StructuredTool]:
    ws = settings.resolved_workspace()
    cmd_timeout = float(settings.shell_timeout_sec)

    def compile_cuda(
        nvcc_executable: str,
        source_relative: str,
        output_binary_relative: str,
    ) -> str:
        _LOG.debug(
            "compile_cuda nvcc=%r src=%r out=%r",
            nvcc_executable,
            source_relative,
            output_binary_relative,
        )
        src = assert_under_workspace(ws, source_relative)
        out = assert_under_workspace(ws, output_binary_relative)
        nvcc = nvcc_executable.strip() or (settings.nvcc_path or "nvcc")
        compile_cuda_source(
            nvcc=nvcc,
            source=src,
            output_binary=out,
            cwd=ws,
            timeout_sec=cmd_timeout,
        )
        return f"编译成功: {out}"

    def run_binary(binary_relative: str, args: str = "") -> str:
        binary = assert_under_workspace(ws, binary_relative)
        argv = args.split() if args.strip() else []
        stdout = run_cuda_binary(binary, argv, cwd=ws, timeout_sec=cmd_timeout)
        return stdout

    def run_ncu(
        ncu_executable: str,
        binary_relative: str,
        program_args: str = "",
        metrics: str | None = None,
    ) -> str:
        binary = assert_under_workspace(ws, binary_relative)
        argv = program_args.split() if program_args.strip() else []
        ncu = ncu_executable.strip() or (settings.ncu_path or "ncu")
        report = profile_with_ncu(
            ncu=ncu,
            binary=binary,
            program_args=argv,
            metrics=metrics,
            cwd=ws,
            timeout_sec=cmd_timeout,
        )
        if len(report) > settings.max_tool_output_chars:
            half = settings.max_tool_output_chars // 2
            report = report[:half] + "\n...[truncated]...\n" + report[-half:]
        return report

    return [
        StructuredTool.from_function(
            name="compile_cuda",
            description="调用 nvcc 编译工作区内的 .cu 源文件为可执行文件。需先自行探测 nvcc 路径。",
            args_schema=CompileCudaArgs,
            func=lambda nvcc_executable, source_relative, output_binary_relative: compile_cuda(
                nvcc_executable, source_relative, output_binary_relative
            ),
        ),
        StructuredTool.from_function(
            name="run_cuda_binary",
            description="运行工作区内已编译的 CUDA 可执行文件。",
            args_schema=RunBinaryArgs,
            func=lambda binary_relative, program_args="": run_binary(binary_relative, program_args),
        ),
        StructuredTool.from_function(
            name="run_ncu_profile",
            description="对二进制运行 ncu 并返回 CSV 文本。需先自行探测 ncu 路径。",
            args_schema=RunNcuArgs,
            func=lambda ncu_executable, binary_relative, program_args="", metrics=None: run_ncu(
                ncu_executable, binary_relative, program_args, metrics
            ),
        ),
    ]
