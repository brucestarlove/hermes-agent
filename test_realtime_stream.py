"""End-to-end tests against the FastAPI app with a synthetic emulator.

These tests verify that the free-running ``EmulatorRuntime`` integrates
correctly with the HTTP layer: the world ticks regardless of /action, the
MJPEG stream advances during idle, and concurrent /state reads land while a
long /action is in flight.
"""

import threading
import time

from fastapi.testclient import TestClient
from PIL import Image

import pokemon_agent.server as server
from pokemon_agent.runtime import EmulatorRuntime


class FakeEmulator:
    """Synthetic emulator with a per-frame-changing screen.

    The pixel at (0, 0) cycles its red channel each frame so consecutive
    PNG encodes produce different bytes — useful for proving the stream is
    actually advancing rather than serving a cached frame.
    """

    def __init__(self):
        self.frame_count = 0
        self.history = []  # list of (event, button, frame)
        self.held = set()

    def get_screen(self):
        img = Image.new("RGB", (160, 144), (255, 255, 255))
        # Encode frame_count into top-left pixel so each tick produces a
        # distinct PNG.
        img.putpixel((0, 0), (self.frame_count % 256, (self.frame_count // 256) % 256, 0))
        return img

    def tick(self, frames=1):
        for _ in range(frames):
            self.frame_count += 1

    def press(self, button, frames=1):
        self.press_button(button)
        self.tick(frames)
        self.release_button(button)

    def press_button(self, button):
        self.held.add(button)
        self.history.append(("press", button, self.frame_count))

    def release_button(self, button):
        self.held.discard(button)
        self.history.append(("release", button, self.frame_count))

    def release_all(self):
        self.held.clear()

    def save_state(self, path):
        pass

    def load_state(self, path):
        pass


def _install_runtime():
    """Replace server._runtime with a FakeEmulator-backed runtime, return old."""
    old = server._runtime
    fake = FakeEmulator()
    runtime = EmulatorRuntime(fake, reader=None, target_hz=120, publish_every=2)
    runtime.start()
    server._runtime = runtime
    return old, runtime


def _restore_runtime(old, runtime):
    runtime.stop(timeout=2.0)
    server._runtime = old


def test_mjpeg_stream_returns_multipart_png_frame():
    old, runtime = _install_runtime()
    try:
        client = TestClient(server.app)
        response = client.get("/stream.mjpg?fps=30&frames=1")
    finally:
        _restore_runtime(old, runtime)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
    assert b"--frame" in response.content
    assert b"Content-Type: image/png" in response.content
    assert b"\x89PNG\r\n\x1a\n" in response.content


def test_mjpeg_stream_advances_during_idle():
    """Without sending any /action, the stream should still produce changing frames."""
    old, runtime = _install_runtime()
    try:
        # Give the runtime a moment to publish a couple of frames.
        time.sleep(0.2)
        client = TestClient(server.app)
        response = client.get("/stream.mjpg?fps=20&frames=3")
    finally:
        _restore_runtime(old, runtime)

    assert response.status_code == 200
    body = response.content
    # The body contains 3 PNG payloads. Slice them out by the PNG magic.
    pngs = []
    cursor = 0
    magic = b"\x89PNG\r\n\x1a\n"
    while True:
        idx = body.find(magic, cursor)
        if idx < 0:
            break
        # End-of-PNG: IEND chunk + CRC = b"IEND\xae\x42\x60\x82"
        end = body.find(b"IEND\xae\x42\x60\x82", idx)
        if end < 0:
            break
        pngs.append(body[idx : end + 8])
        cursor = end + 8
    assert len(pngs) >= 2, f"expected ≥2 PNGs, got {len(pngs)}"
    # At least two must differ — proves the world ticked between frames.
    assert any(pngs[i] != pngs[0] for i in range(1, len(pngs))), "frames did not change"


def test_action_drives_runtime_and_returns_state():
    """POST /action with a no-op `wait_5` should succeed and report state."""
    old, runtime = _install_runtime()
    try:
        client = TestClient(server.app)
        response = client.post("/action", json={"actions": ["wait_5"]})
    finally:
        _restore_runtime(old, runtime)

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["actions_executed"] == 1


def test_state_responsive_during_long_action():
    """While /action is in flight, /state must still respond promptly."""
    old, runtime = _install_runtime()
    state_latency = {}
    try:
        client = TestClient(server.app)

        # Fire a long-running action in a background thread.
        action_done = threading.Event()

        def post_long_action():
            client.post("/action", json={"actions": ["wait_120"]})
            action_done.set()

        t = threading.Thread(target=post_long_action, daemon=True)
        t.start()

        # Give the action a moment to start ticking.
        time.sleep(0.1)
        start = time.perf_counter()
        r = client.get("/state")
        state_latency["seconds"] = time.perf_counter() - start
        assert r.status_code == 200

        action_done.wait(timeout=5.0)
        t.join(timeout=1.0)
    finally:
        _restore_runtime(old, runtime)

    # /state during a 120-frame wait should return well under a second.
    assert state_latency["seconds"] < 0.5, (
        f"/state was blocked: {state_latency['seconds']:.3f}s"
    )


def test_walk_action_emits_press_and_release_edges():
    """A `walk_up` should produce one press+release edge pair on the fake emu."""
    old, runtime = _install_runtime()
    try:
        client = TestClient(server.app)
        r = client.post("/action", json={"actions": ["walk_up"]})
        assert r.status_code == 200
        emu = runtime._emu
        ups_pressed = [h for h in emu.history if h[0] == "press" and h[1] == "up"]
        ups_released = [h for h in emu.history if h[0] == "release" and h[1] == "up"]
        assert len(ups_pressed) == 1
        assert len(ups_released) == 1
        assert "up" not in emu.held
    finally:
        _restore_runtime(old, runtime)
