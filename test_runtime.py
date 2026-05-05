"""Unit tests for ``pokemon_agent.runtime.EmulatorRuntime``.

Drives the runtime directly with a synthetic emulator so we can assert
timing, input scheduling, save/load barriers, and clean shutdown without a
real ROM.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import TimeoutError as FutureTimeout

import pytest
from PIL import Image

from pokemon_agent.runtime import (
    AUntilDialogEndProgram,
    EmulatorRuntime,
    InputEdge,
    ScheduledProgram,
)


class FakeEmulator:
    """Minimal emulator stand-in for runtime tests."""

    def __init__(self) -> None:
        self.frame_count = 0
        self.history: list[tuple[str, str, int]] = []  # (event, button, frame)
        self.held: set[str] = set()
        self.released_all_count = 0
        self.saved: list[str] = []
        self.loaded: list[str] = []
        self._lock = threading.Lock()

    def tick(self, frames: int = 1) -> None:
        for _ in range(frames):
            self.frame_count += 1

    def press(self, button: str, frames: int = 1) -> None:
        self.press_button(button)
        self.tick(frames)
        self.release_button(button)

    def press_button(self, button: str) -> None:
        with self._lock:
            self.held.add(button)
            self.history.append(("press", button, self.frame_count))

    def release_button(self, button: str) -> None:
        with self._lock:
            self.held.discard(button)
            self.history.append(("release", button, self.frame_count))

    def release_all(self) -> None:
        with self._lock:
            self.held.clear()
            self.released_all_count += 1

    def get_screen(self) -> Image.Image:
        # Return a tiny image whose pixel changes per frame so consecutive
        # publications differ — useful for the MJPEG advancement test.
        img = Image.new("RGB", (4, 4), (0, 0, 0))
        img.putpixel((0, 0), (self.frame_count % 256, 0, 0))
        return img

    def save_state(self, path: str) -> None:
        self.saved.append(path)

    def load_state(self, path: str) -> None:
        self.loaded.append(path)


class FakeDialogReader:
    """Reader stub whose ``read_dialog`` flips to inactive after N calls."""

    def __init__(self, *, active_for_calls: int = 2) -> None:
        self._calls = 0
        self._active_for = active_for_calls

    def read_dialog(self) -> dict:
        self._calls += 1
        return {"active": self._calls <= self._active_for, "text": ""}

    # build_game_state expects these too — return minimal data so it doesn't crash.
    def read_player(self) -> dict: return {}
    def read_party(self) -> list: return []
    def read_bag(self) -> dict: return {}
    def read_battle(self) -> dict: return {}
    def read_map_info(self) -> dict: return {}
    def read_flags(self) -> dict: return {}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_tick_rate_within_tolerance():
    """Free-running runtime should hit ~60 Hz (allow ±25% on noisy CI/WSL)."""
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=60, publish_every=0)
    rt.start()
    try:
        time.sleep(1.0)
        ticks = emu.frame_count
    finally:
        rt.stop(timeout=2.0)
    assert 45 <= ticks <= 75, f"expected ~60 Hz over 1s, got {ticks} ticks"


def test_shutdown_joins_thread_cleanly():
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=60, publish_every=0)
    rt.start()
    time.sleep(0.05)
    rt.stop(timeout=1.0)
    assert rt._thread is not None
    assert not rt._thread.is_alive()


# ---------------------------------------------------------------------------
# Input programs
# ---------------------------------------------------------------------------

def test_scheduled_program_press_and_release_at_correct_offsets():
    """A ``walk_up``-style program should press at offset 0, release at offset 8."""
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=120, publish_every=0)
    rt.start()
    try:
        prog = ScheduledProgram(
            edges=[InputEdge(0, "up", True), InputEdge(8, "up", False)],
            duration=20,
        )
        future = rt.submit_program(prog)
        result = future.result(timeout=2.0)
        assert "frame_count" in result
    finally:
        rt.stop(timeout=2.0)

    # Find the press and release in history; release should come ≥ 8 frames after press.
    press_events = [h for h in emu.history if h[0] == "press" and h[1] == "up"]
    release_events = [h for h in emu.history if h[0] == "release" and h[1] == "up"]
    assert len(press_events) == 1
    assert len(release_events) == 1
    press_frame = press_events[0][2]
    release_frame = release_events[0][2]
    assert release_frame - press_frame >= 8
    # 'up' should not still be held after the program completes
    assert "up" not in emu.held


def test_wait_program_completes_after_n_frames():
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=120, publish_every=0)
    rt.start()
    try:
        start_frame = emu.frame_count
        prog = ScheduledProgram(edges=[], duration=30)
        future = rt.submit_program(prog)
        future.result(timeout=2.0)
        elapsed = emu.frame_count - start_frame
    finally:
        rt.stop(timeout=2.0)
    assert elapsed >= 30


def test_programs_run_sequentially_when_pipelined():
    """Submitting two programs back-to-back: the second waits for the first."""
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=240, publish_every=0)
    rt.start()
    try:
        prog_a = ScheduledProgram(
            edges=[InputEdge(0, "a", True), InputEdge(2, "a", False)],
            duration=5,
        )
        prog_b = ScheduledProgram(
            edges=[InputEdge(0, "b", True), InputEdge(2, "b", False)],
            duration=5,
        )
        fut_a = rt.submit_program(prog_a)
        fut_b = rt.submit_program(prog_b)
        fut_a.result(timeout=2.0)
        fut_b.result(timeout=2.0)
    finally:
        rt.stop(timeout=2.0)

    a_press = next(h for h in emu.history if h == ("press", "a", h[2]))
    b_press = next(h for h in emu.history if h[0] == "press" and h[1] == "b")
    a_release = next(h for h in emu.history if h[0] == "release" and h[1] == "a")
    # b should press only AFTER a has been released (sequential queue).
    assert b_press[2] >= a_release[2]


# ---------------------------------------------------------------------------
# Save / load barrier
# ---------------------------------------------------------------------------

def test_save_state_executes_inline_with_loop():
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=120, publish_every=0)
    rt.start()
    try:
        fut = rt.save_state("/tmp/test-runtime-save.state")
        fut.result(timeout=2.0)
    finally:
        rt.stop(timeout=2.0)
    assert "/tmp/test-runtime-save.state" in emu.saved


def test_load_releases_held_buttons():
    """Loading a state mid-program must release any held buttons (no phantoms)."""
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=240, publish_every=0)
    rt.start()
    try:
        # Hold 'right' for a long time.
        prog = ScheduledProgram(
            edges=[InputEdge(0, "right", True), InputEdge(500, "right", False)],
            duration=500,
        )
        fut_prog = rt.submit_program(prog)
        # Give it a moment to start pressing.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and "right" not in emu.held:
            time.sleep(0.005)
        assert "right" in emu.held, "press_button should have run on the runtime thread"

        # Now load a state — runtime should cancel the program and release_all.
        fut_load = rt.load_state("/tmp/test-runtime-load.state")
        fut_load.result(timeout=2.0)

        # Program future should be cancelled/failed (cancel-by-load).
        with pytest.raises(Exception):
            fut_prog.result(timeout=0.5)
    finally:
        rt.stop(timeout=2.0)

    assert "right" not in emu.held
    assert emu.released_all_count >= 1
    assert "/tmp/test-runtime-load.state" in emu.loaded


# ---------------------------------------------------------------------------
# A-until-dialog-end
# ---------------------------------------------------------------------------

def test_a_until_dialog_end_completes_when_dialog_clears():
    """The program should stop after dialog flips inactive within MAX_INTERVALS."""
    emu = FakeEmulator()
    reader = FakeDialogReader(active_for_calls=2)  # active for 2 polls then clears
    rt = EmulatorRuntime(emu, reader=reader, target_hz=240, publish_every=0)
    rt.start()
    try:
        prog = AUntilDialogEndProgram()
        fut = rt.submit_program(prog)
        fut.result(timeout=3.0)  # must complete within MAX_INTERVALS (10*30 frames)
    finally:
        rt.stop(timeout=2.0)
    # We should have seen at least one A press
    a_presses = [h for h in emu.history if h[0] == "press" and h[1] == "a"]
    assert len(a_presses) >= 1


# ---------------------------------------------------------------------------
# Frame publishing
# ---------------------------------------------------------------------------

def test_published_frame_advances_during_idle():
    """With no /action sent, peek_frame seq must still increase as ticks happen."""
    emu = FakeEmulator()
    rt = EmulatorRuntime(emu, reader=None, target_hz=120, publish_every=2)
    rt.start()
    try:
        # Wait for at least the initial publish.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and rt.peek_frame()[1] == b"":
            time.sleep(0.01)
        seq_a, png_a = rt.peek_frame()
        assert png_a, "initial frame should be published"
        # Wait for a few more publishes.
        time.sleep(0.3)
        seq_b, png_b = rt.peek_frame()
    finally:
        rt.stop(timeout=2.0)
    assert seq_b > seq_a, f"seq did not advance during idle ({seq_a} → {seq_b})"
