"""Tests for the bundled PeonPing lifecycle plugin."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


class DummyContext:
    def __init__(self):
        self.hooks = []
        self.commands = {}

    def register_hook(self, name, callback):
        self.hooks.append((name, callback))

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {
            "handler": handler,
            "description": description,
            "args_hint": args_hint,
        }


def test_manifest_declares_standalone_lifecycle_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load((repo_root / "plugins" / "peonping" / "plugin.yaml").read_text())

    assert manifest["name"] == "peonping"
    assert manifest["kind"] == "standalone"
    assert "pre_tool_call" in manifest["hooks"]
    assert "subagent_stop" in manifest["hooks"]
    assert "on_session_finalize" in manifest["hooks"]
    assert "on_session_reset" in manifest["hooks"]
    assert "on_session_end" not in manifest["hooks"]
    assert "peonping" in manifest["commands"]


def test_default_config_path_uses_hermes_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_PEONPING_CONFIG", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    from plugins.peonping.config import default_config_path

    assert default_config_path() == tmp_path / "hermes-home" / "peonping" / "config.json"


def test_mapping_adds_voicepack_hint_and_session_prefix():
    from plugins.peonping.config import AdapterConfig
    from plugins.peonping.mapper import map_hermes_payload

    event = map_hermes_payload(
        {
            "hook_event_name": "pre_llm_call",
            "session_id": "sess-1",
            "cwd": "/repo",
            "extra": {"is_first_turn": True},
        },
        AdapterConfig(voicepack="test-pack"),
    )

    assert event["hook_event_name"] == "SessionStart"
    assert event["session_id"] == "hermes-sess-1"
    assert event["cwd"] == "/repo"
    assert event["source"] == "hermes"
    assert event["voicepack"] == "test-pack"


def test_selected_pre_tool_call_maps_to_progress_notification():
    from plugins.peonping.config import AdapterConfig
    from plugins.peonping.mapper import map_hermes_payload

    event = map_hermes_payload(
        {
            "hook_event_name": "pre_tool_call",
            "tool_name": "terminal",
            "tool_input": {"command": "python -m pytest tests/plugins/test_peonping_plugin.py"},
            "session_id": "sess-1",
            "cwd": "/repo",
            "extra": {},
        },
        AdapterConfig(tool_progress_events=["terminal"]),
    )

    assert event["hook_event_name"] == "Notification"
    assert event["notification_type"] == "progress"
    assert event["cesp_category"] == "task.progress"
    assert event["tool_name"] == "terminal"
    assert "pytest" in event["message"]


def test_terminal_failure_maps_to_post_tool_use_failure():
    from plugins.peonping.config import AdapterConfig
    from plugins.peonping.mapper import map_hermes_payload

    event = map_hermes_payload(
        {
            "hook_event_name": "post_tool_call",
            "tool_name": "terminal",
            "tool_input": {"command": "false"},
            "session_id": "sess-1",
            "cwd": "/repo",
            "extra": {"result": json.dumps({"output": "", "exit_code": 1})},
        },
        AdapterConfig(tool_error_events=["terminal"]),
    )

    assert event["hook_event_name"] == "PostToolUseFailure"
    assert event["tool_name"] == "Bash"
    assert event["hermes_tool_name"] == "terminal"
    assert "exit code 1" in event["error"].lower()


def test_subagent_stop_maps_completion_status_and_summary():
    from plugins.peonping.config import AdapterConfig
    from plugins.peonping.mapper import map_hermes_payload

    event = map_hermes_payload(
        {
            "hook_event_name": "subagent_stop",
            "session_id": "parent-sess",
            "cwd": "/repo",
            "extra": {
                "child_status": "completed",
                "child_summary": "Implemented the feature and updated tests.",
            },
        },
        AdapterConfig(),
    )

    assert event["hook_event_name"] == "SubagentStop"
    assert event["cesp_category"] == "subagent.completed"
    assert event["status"] == "completed"
    assert "Implemented the feature" in event["summary"]


def test_plugin_registers_hooks_and_status_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_PEONPING_CONFIG", str(tmp_path / "config.json"))

    from plugins import peonping
    from plugins.peonping.config import AdapterConfig, save_config

    save_config(AdapterConfig(voicepack="test-pack"), tmp_path / "config.json")
    ctx = DummyContext()

    peonping.register(ctx)

    hook_names = [name for name, _ in ctx.hooks]
    assert "pre_tool_call" in hook_names
    assert "subagent_stop" in hook_names
    assert "on_session_finalize" in hook_names
    assert "on_session_reset" in hook_names
    assert "on_session_end" not in hook_names
    assert "peonping" in ctx.commands
    status = ctx.commands["peonping"]["handler"]("")
    assert "PeonPing adapter" in status
    assert "test-pack" in status


def test_argv_preserves_existing_command_path_with_spaces(tmp_path):
    from plugins.peonping.adapter import _argv

    command_path = tmp_path / "dir with spaces" / "peon"
    command_path.parent.mkdir()
    command_path.write_text("#!/bin/sh\n", encoding="utf-8")

    assert _argv(str(command_path)) == [str(command_path)]


def test_emit_payload_returns_error_when_peon_command_cannot_start():
    from plugins.peonping.adapter import emit_payload
    from plugins.peonping.config import AdapterConfig

    result = emit_payload(
        {
            "hook_event_name": "post_llm_call",
            "session_id": "sess-1",
            "cwd": "/repo",
            "extra": {},
        },
        AdapterConfig(peon_command="/definitely/missing/peon"),
    )

    assert result.returncode != 0
    assert result.skipped is False
    assert "failed to run" in result.stderr.lower()


def test_emit_payload_dry_run_prints_mapped_event(capsys):
    from plugins.peonping.adapter import emit_payload
    from plugins.peonping.config import AdapterConfig

    result = emit_payload(
        {
            "hook_event_name": "post_llm_call",
            "session_id": "sess-1",
            "cwd": "/repo",
            "extra": {},
        },
        AdapterConfig(),
        dry_run=True,
    )

    assert result.returncode == 0
    assert result.skipped is False
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["hook_event_name"] == "Stop"
