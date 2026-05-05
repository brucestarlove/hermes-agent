"""Emulator runtime — owns a dedicated 60 Hz tick thread.

The runtime thread is the *only* thread that touches the emulator. FastAPI
request handlers communicate with it via:

  * a ``queue.Queue`` of typed commands (input programs, save/load, shutdown)
  * a ``threading.Lock`` for foreign-thread snapshot reads
  * a published latest-frame slot under its own small lock

The world ticks continuously, independent of agent activity.  Dashboard
viewers see NPC animation, dialog text typing, and battle effects play at
real time — not only when ``/action`` is in flight.
"""

from __future__ import annotations

import io
import logging
import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Deque, List, Optional, Tuple

if TYPE_CHECKING:
    from pokemon_agent.emulator import Emulator

log = logging.getLogger("pokemon_agent.runtime")


# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

@dataclass(order=True)
class InputEdge:
    """A press or release edge at a frame *offset* from program start."""
    offset: int
    button: str = field(compare=False)
    press: bool = field(compare=False)  # True = press, False = release


class Program:
    """Base class. ``advance()`` is called once per frame; True ⇒ done."""

    def __init__(self) -> None:
        self.future: Future = Future()
        # Mark as running so callers can't cancel it via Future.cancel(); we
        # control completion explicitly by calling set_result/set_exception.
        self.future.set_running_or_notify_cancel()
        self._start_frame: int = -1

    def advance(self, runtime: "EmulatorRuntime") -> bool:
        raise NotImplementedError

    def fail(self, exc: BaseException) -> None:
        if not self.future.done():
            self.future.set_exception(exc)

    def complete(self, result: Any) -> None:
        if not self.future.done():
            self.future.set_result(result)


class ScheduledProgram(Program):
    """Apply a list of frame-relative edges over a fixed duration.

    Used for ``press_X``, ``walk_X``, ``hold_X_N``, ``wait_N`` actions.
    """

    def __init__(self, edges: List[InputEdge], duration: int) -> None:
        super().__init__()
        self._edges: List[InputEdge] = sorted(edges)
        self._duration = max(0, int(duration))
        self._idx = 0

    def advance(self, runtime: "EmulatorRuntime") -> bool:
        if self._start_frame < 0:
            self._start_frame = runtime.frame_count
        elapsed = runtime.frame_count - self._start_frame
        while self._idx < len(self._edges) and self._edges[self._idx].offset <= elapsed:
            edge = self._edges[self._idx]
            if edge.press:
                runtime._emu.press_button(edge.button)
            else:
                runtime._emu.release_button(edge.button)
            self._idx += 1
        return elapsed >= self._duration


class AUntilDialogEndProgram(Program):
    """Press A every 30 frames until ``dialog.active`` is False or 300 frames elapse.

    Mirrors the legacy server behavior at server.py:193–204 but runs on the
    loop thread, so the dialog poll is in-process and consistent with the tick.
    """

    INTERVAL = 30
    MAX_INTERVALS = 10
    PRESS_HOLD_FRAMES = 2  # how many frames A is held per tap

    def __init__(self) -> None:
        super().__init__()
        self._next_press_at = 1     # press on the first iteration boundary
        self._release_at = -1
        self._intervals_done = 0

    def advance(self, runtime: "EmulatorRuntime") -> bool:
        if self._start_frame < 0:
            self._start_frame = runtime.frame_count
        elapsed = runtime.frame_count - self._start_frame

        # Release A if its hold window has elapsed.
        if self._release_at >= 0 and elapsed >= self._release_at:
            try:
                runtime._emu.release_button("a")
            except Exception:
                pass
            self._release_at = -1

        # On each interval boundary, check dialog state and (if still active) press A.
        if elapsed >= self._next_press_at:
            if runtime._reader is not None:
                try:
                    from pokemon_agent.state.builder import build_game_state
                    state = build_game_state(runtime._reader)
                    dialog = state.get("dialog") or {}
                    if not dialog.get("active", False):
                        return True
                except Exception:
                    pass
            self._intervals_done += 1
            if self._intervals_done > self.MAX_INTERVALS:
                return True
            try:
                runtime._emu.press_button("a")
            except Exception:
                pass
            self._release_at = elapsed + self.PRESS_HOLD_FRAMES
            self._next_press_at = elapsed + self.INTERVAL

        return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

# Command tuples stored in the queue: (kind, payload)
# kind ∈ { "RUN", "SAVE", "LOAD", "SHUTDOWN" }
#   RUN      payload = Program
#   SAVE     payload = (path: str, future: Future)
#   LOAD     payload = (path: str, future: Future)
#   SHUTDOWN payload = None


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class EmulatorRuntime:
    """Owns the emulator + a 60 Hz tick thread.

    Public API is thread-safe.  All methods may be called from the asyncio
    worker thread; the runtime thread itself is the only thread that touches
    the underlying emulator object.
    """

    def __init__(
        self,
        emu: "Emulator",
        reader: Optional[Any],
        *,
        target_hz: int = 60,
        publish_every: int = 2,
    ) -> None:
        self._emu = emu
        self._reader = reader
        self._target_hz = max(1, int(target_hz))
        self._publish_every = max(0, int(publish_every))
        self._target_dt = 1.0 / self._target_hz

        # Coordination
        self._lock = threading.Lock()           # foreign-thread emulator access
        self._frame_lock = threading.Lock()     # latest-frame slot
        self._cmd_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._shutdown = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Program state (touched only by the runtime thread)
        self._pending_programs: Deque[Program] = deque()
        self._active_program: Optional[Program] = None

        # Latest frame slot (PNG bytes + monotonic seq)
        self._latest_png: bytes = b""
        self._latest_seq: int = 0

        # Drift telemetry
        self._behind_warnings = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._shutdown.clear()
        self._thread = threading.Thread(
            target=self._loop, name="pokemon-runtime", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._cmd_queue.put(("SHUTDOWN", None))
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        # Fail any unfinished programs so awaiting handlers don't hang.
        if self._active_program is not None:
            self._active_program.fail(RuntimeError("runtime shut down"))
            self._active_program = None
        while self._pending_programs:
            self._pending_programs.popleft().fail(RuntimeError("runtime shut down"))

    # -- handler-facing API ------------------------------------------------

    @property
    def frame_count(self) -> int:
        return self._emu.frame_count

    def submit_program(self, program: Program) -> Future:
        """Queue an input program; returns its Future (resolved when complete)."""
        if self._shutdown.is_set():
            program.fail(RuntimeError("runtime not running"))
            return program.future
        self._cmd_queue.put(("RUN", program))
        return program.future

    def save_state(self, path: str) -> Future:
        """Enqueue a save-state operation; resolves when written."""
        fut: Future = Future()
        fut.set_running_or_notify_cancel()
        self._cmd_queue.put(("SAVE", (path, fut)))
        return fut

    def load_state(self, path: str) -> Future:
        """Enqueue a load-state operation; cancels in-flight programs."""
        fut: Future = Future()
        fut.set_running_or_notify_cancel()
        self._cmd_queue.put(("LOAD", (path, fut)))
        return fut

    def snapshot_state(self) -> dict:
        """Lock-held atomic state snapshot. Synchronous; safe from any thread."""
        from pokemon_agent.state.builder import build_game_state
        if self._reader is None:
            return {"metadata": {"frame_count": self._emu.frame_count}}
        with self._lock:
            return build_game_state(self._reader, frame_count=self._emu.frame_count)

    def peek_frame(self) -> Tuple[int, bytes]:
        """Return ``(seq, png_bytes)`` of the most-recently-published frame."""
        with self._frame_lock:
            return self._latest_seq, self._latest_png

    # -- the loop ----------------------------------------------------------

    def _loop(self) -> None:
        next_deadline = time.monotonic()
        # Make sure we publish a frame as soon as we start so peek_frame()
        # has something to return before the first tick interval elapses.
        self._publish_initial_frame()

        while not self._shutdown.is_set():
            screen_to_publish = None
            with self._lock:
                if self._shutdown.is_set():
                    break
                self._drain_commands()
                if self._shutdown.is_set():
                    break
                self._advance_active_program()
                try:
                    self._emu.tick(1)
                except Exception:
                    log.exception("emulator tick failed; runtime stopping")
                    break
                if (
                    self._publish_every > 0
                    and self._emu.frame_count % self._publish_every == 0
                ):
                    try:
                        screen_to_publish = self._emu.get_screen()
                    except Exception:
                        screen_to_publish = None

            if screen_to_publish is not None:
                self._encode_and_publish(screen_to_publish)

            next_deadline += self._target_dt
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            elif sleep_for < -0.25:
                # Fell badly behind (e.g. paused for save/load or system stall).
                # Re-anchor instead of trying to catch up.
                self._behind_warnings += 1
                if self._behind_warnings <= 3 or self._behind_warnings % 60 == 0:
                    log.warning(
                        "runtime fell behind by %.0fms — re-anchoring deadline",
                        -sleep_for * 1000,
                    )
                next_deadline = time.monotonic()

    # -- loop helpers (runtime thread only) --------------------------------

    def _drain_commands(self) -> None:
        while True:
            try:
                kind, payload = self._cmd_queue.get_nowait()
            except queue.Empty:
                return
            if kind == "RUN":
                self._pending_programs.append(payload)
            elif kind == "SAVE":
                path, fut = payload
                try:
                    self._release_all_buttons()
                    self._emu.save_state(path)
                    fut.set_result(None)
                except Exception as e:
                    fut.set_exception(e)
            elif kind == "LOAD":
                path, fut = payload
                try:
                    # Cancel anything in flight — load is a fresh world.
                    self._cancel_all_programs(reason="load")
                    self._release_all_buttons()
                    self._emu.load_state(path)
                    fut.set_result(None)
                except Exception as e:
                    fut.set_exception(e)
            elif kind == "SHUTDOWN":
                self._shutdown.set()
                return

    def _advance_active_program(self) -> None:
        if self._active_program is None:
            if not self._pending_programs:
                return
            self._active_program = self._pending_programs.popleft()
        try:
            done = self._active_program.advance(self)
        except Exception as e:
            self._active_program.fail(e)
            self._active_program = None
            return
        if done:
            self._active_program.complete({"frame_count": self._emu.frame_count})
            self._active_program = None

    def _release_all_buttons(self) -> None:
        try:
            self._emu.release_all()
        except Exception:
            pass

    def _cancel_all_programs(self, *, reason: str) -> None:
        if self._active_program is not None:
            self._active_program.fail(RuntimeError(f"cancelled by {reason}"))
            self._active_program = None
        while self._pending_programs:
            self._pending_programs.popleft().fail(RuntimeError(f"cancelled by {reason}"))

    def _encode_and_publish(self, screen: Any) -> None:
        """Encode a PIL frame as PNG and publish it to the latest-frame slot."""
        try:
            from PIL import Image
            buf = io.BytesIO()
            if not isinstance(screen, Image.Image):
                import numpy as np  # only needed for the array path
                screen = Image.fromarray(screen)
            screen.save(buf, format="PNG")
            png = buf.getvalue()
        except Exception:
            log.exception("frame encode failed")
            return
        with self._frame_lock:
            self._latest_seq += 1
            self._latest_png = png

    def _publish_initial_frame(self) -> None:
        try:
            with self._lock:
                screen = self._emu.get_screen()
        except Exception:
            return
        self._encode_and_publish(screen)
