from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DOTENV_CANDIDATES = (Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env")


class AppSettings(BaseSettings):
    """统一配置入口：仅从本类读取，业务代码禁止散落 os.getenv。"""

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    environment: Literal["development", "production"] = Field(
        default="production",
        validation_alias=AliasChoices("HW_PROBE_ENVIRONMENT", "ENVIRONMENT"),
    )
    api_key: str = Field(
        default="",
        validation_alias=AliasChoices("API_KEY"),
    )
    base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("BASE_URL"),
    )
    model: str = Field(
        default="gpt-4o",
        validation_alias=AliasChoices("BASE_MODEL", "MODEL"),
    )

    workspace_root: Path = Field(
        default=Path("/workspace"),
        validation_alias=AliasChoices("HW_PROBE_WORKSPACE_ROOT"),
    )
    target_spec_path: Path = Field(
        default=Path("/target/target_spec.json"),
        validation_alias=AliasChoices("HW_PROBE_TARGET_SPEC_PATH"),
    )
    build_dir_name: str = Field(
        default="build",
        validation_alias=AliasChoices("HW_PROBE_BUILD_DIR_NAME"),
    )
    probes_dir_name: str = Field(
        default="probes",
        validation_alias=AliasChoices("HW_PROBE_PROBES_DIR_NAME"),
    )

    nvcc_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HW_PROBE_NVCC_PATH"),
    )
    ncu_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("HW_PROBE_NCU_PATH"),
    )

    shell_timeout_sec: int = Field(
        default=120,
        ge=1,
        validation_alias=AliasChoices("HW_PROBE_SHELL_TIMEOUT_SEC"),
    )
    max_tool_output_chars: int = Field(
        default=120_000,
        ge=1_000,
        validation_alias=AliasChoices("HW_PROBE_MAX_TOOL_OUTPUT_CHARS"),
    )
    react_max_steps: int = Field(
        default=24,
        ge=1,
        validation_alias=AliasChoices("HW_PROBE_REACT_MAX_STEPS"),
    )
    supervisor_max_loops: int = Field(
        default=12,
        ge=1,
        validation_alias=AliasChoices("HW_PROBE_SUPERVISOR_MAX_LOOPS"),
    )
    max_total_runtime_minutes: int = Field(
        default=30,
        ge=1,
        le=480,
        validation_alias=AliasChoices("HW_PROBE_MAX_TOTAL_RUNTIME_MINUTES"),
    )

    output_filename: str = Field(
        default="output.json",
        validation_alias=AliasChoices("HW_PROBE_OUTPUT_FILENAME"),
    )
    results_log_name: str = Field(
        default="results.log",
        validation_alias=AliasChoices("HW_PROBE_RESULTS_LOG_NAME"),
    )

    log_dir_name: str = Field(
        default="log",
        validation_alias=AliasChoices("HW_PROBE_LOG_DIR_NAME"),
    )
    console_log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("HW_PROBE_CONSOLE_LOG_LEVEL"),
    )
    file_log_level: str = Field(
        default="DEBUG",
        validation_alias=AliasChoices("HW_PROBE_FILE_LOG_LEVEL"),
    )
    debug_log_filename: str = Field(
        default="hw_probe.debug.log",
        validation_alias=AliasChoices("HW_PROBE_DEBUG_LOG_FILENAME"),
    )
    llm_trace_jsonl_filename: str = Field(
        default="llm_trace.jsonl",
        validation_alias=AliasChoices("HW_PROBE_LLM_TRACE_JSONL"),
    )
    llm_session_markdown_filename: str = Field(
        default="llm_session.md",
        validation_alias=AliasChoices("HW_PROBE_LLM_SESSION_MARKDOWN"),
    )
    llm_session_markdown_max_lines: int = Field(
        default=3000,
        ge=1,
        le=500_000,
        validation_alias=AliasChoices("HW_PROBE_LLM_SESSION_MARKDOWN_MAX_LINES"),
    )
    log_file_max_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=256 * 1024,
        validation_alias=AliasChoices("HW_PROBE_LOG_FILE_MAX_BYTES"),
    )
    log_file_backup_count: int = Field(
        default=5,
        ge=1,
        le=32,
        validation_alias=AliasChoices("HW_PROBE_LOG_FILE_BACKUP_COUNT"),
    )

    @field_validator("workspace_root", "target_spec_path", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        if isinstance(v, Path):
            return v
        return Path(str(v))

    @model_validator(mode="after")
    def _development_use_repo_relative_paths(self) -> AppSettings:
        """development 下将评测容器默认绝对路径改为仓库相对路径（便于本地从项目根运行）。"""
        if self.environment != "development":
            return self
        default_target = Path("/target/target_spec.json")
        default_ws = Path("/workspace")
        if self.target_spec_path == default_target:
            self.target_spec_path = Path("target/target_spec.json")
        if self.workspace_root == default_ws:
            self.workspace_root = Path(".")
        return self

    @property
    def build_dir(self) -> Path:
        return self.workspace_root / self.build_dir_name

    @property
    def probes_dir(self) -> Path:
        return self.workspace_root / self.probes_dir_name

    def resolved_workspace(self) -> Path:
        return self.workspace_root.expanduser().resolve()

    def resolved_log_dir(self) -> Path:
        return self.resolved_workspace() / self.log_dir_name

    def validate_llm_config(self) -> None:
        if not self.api_key.strip():
            raise RuntimeError(
                "缺少 API_KEY：请在评测环境注入 API_KEY，或在开发环境于 .env 中配置。"
                "本程序不会伪造凭据或绕过鉴权。"
            )


def _find_dotenv_file() -> Path | None:
    for p in _DOTENV_CANDIDATES:
        if p.is_file():
            return p
    return None


def bootstrap_settings(*, dev: bool = False) -> AppSettings:
    """生产默认不读取 .env；本地使用 ``--dev`` 时加载 .env（不覆盖已注入的环境变量）。"""
    if dev:
        env_file = _find_dotenv_file()
        if env_file is not None:
            load_dotenv(env_file, override=False)
        os.environ.setdefault("HW_PROBE_ENVIRONMENT", "development")
    elif os.environ.get("HW_PROBE_ENVIRONMENT", "").strip().lower() == "development":
        env_file = _find_dotenv_file()
        if env_file is not None:
            load_dotenv(env_file, override=False)
    return AppSettings()
