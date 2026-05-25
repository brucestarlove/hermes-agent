from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1

DEFAULT_ENABLED_EVENTS: Dict[str, bool] = {
    "pre_tool_call": True,
    "pre_llm_call": True,
    "post_llm_call": True,
    "pre_approval_request": True,
    "post_tool_call": True,
    "subagent_stop": True,
    "on_session_finalize": True,
    "on_session_reset": True,
}

DEFAULT_DEBOUNCE_SECONDS: Dict[str, float] = {
    "SessionStart": 30.0,
    "Stop": 1.0,
    "PermissionRequest": 0.25,
    "PostToolUseFailure": 0.25,
    "SubagentStop": 0.5,
}


@dataclass
class AdapterConfig:
    """User controls for Hermes lifecycle events emitted to PeonPing."""

    schema_version: int = SCHEMA_VERSION
    enabled: bool = True
    source: str = "hermes"
    session_prefix: str = "hermes-"
    peon_command: str = ""
    peon_dir: str = ""
    voicepack: str = ""
    enabled_events: Dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_ENABLED_EVENTS))
    tool_error_events: List[str] = field(default_factory=lambda: ["terminal"])
    tool_progress_events: List[str] = field(default_factory=list)
    tool_success_events: List[str] = field(default_factory=list)
    debounce_seconds: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_DEBOUNCE_SECONDS))
    desktop_notifications: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdapterConfig":
        merged_events = dict(DEFAULT_ENABLED_EVENTS)
        raw_events = data.get("enabled_events", {})
        if isinstance(raw_events, dict):
            merged_events.update({str(k): bool(v) for k, v in raw_events.items()})

        merged_debounce = dict(DEFAULT_DEBOUNCE_SECONDS)
        raw_debounce = data.get("debounce_seconds", {})
        if isinstance(raw_debounce, dict):
            for key, value in raw_debounce.items():
                try:
                    merged_debounce[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION),
            enabled=bool(data.get("enabled", True)),
            source=str(data.get("source", "hermes") or "hermes"),
            session_prefix=str(data.get("session_prefix", "hermes-") or ""),
            peon_command=str(data.get("peon_command", "") or ""),
            peon_dir=str(data.get("peon_dir", "") or ""),
            voicepack=str(data.get("voicepack", "") or ""),
            enabled_events=merged_events,
            tool_error_events=_string_list(data.get("tool_error_events"), ["terminal"]),
            tool_progress_events=_string_list(data.get("tool_progress_events"), []),
            tool_success_events=_string_list(data.get("tool_success_events"), []),
            debounce_seconds=merged_debounce,
            desktop_notifications=_optional_bool(data.get("desktop_notifications")),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = SCHEMA_VERSION
        return data

    def event_enabled(self, event_name: str) -> bool:
        return self.enabled and bool(self.enabled_events.get(event_name, False))


def _string_list(value: Any, default: Iterable[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return list(default)


def _optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def default_config_path() -> Path:
    env_path = os.environ.get("HERMES_PEONPING_CONFIG")
    if env_path:
        return Path(env_path).expanduser()

    return get_hermes_home() / "peonping" / "config.json"


def load_config(path: Optional[Path | str] = None) -> AdapterConfig:
    cfg_path = Path(path).expanduser() if path else default_config_path()
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return AdapterConfig()
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid PeonPing adapter config JSON at {cfg_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"PeonPing adapter config at {cfg_path} must be a JSON object")
    return AdapterConfig.from_dict(raw)


def save_config(config: AdapterConfig, path: Optional[Path | str] = None) -> Path:
    cfg_path = Path(path).expanduser() if path else default_config_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return cfg_path
