"""
Pokemon Agent — FastAPI Game Server

Provides HTTP + WebSocket API for controlling a Game Boy / GBA emulator
running a Pokemon ROM, reading game state, and broadcasting events.

The emulator runs in a dedicated background thread (``EmulatorRuntime``)
that ticks at 60 Hz independently of HTTP traffic, so dashboard viewers see
NPC animation, dialog text typing, and battle effects play in real time.
"""

import asyncio
import base64
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from pokemon_agent.runtime import (
    AUntilDialogEndProgram,
    EmulatorRuntime,
    InputEdge,
    Program,
    ScheduledProgram,
)

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GameConfig(BaseModel):
    """Server configuration — set before startup."""
    rom_path: str
    game_type: str = "auto"       # "red", "firered", or "auto"
    port: int = 8765
    data_dir: str = "~/.pokemon-agent"
    load_state: Optional[str] = None  # Save-state name to auto-load on startup


class ActionRequest(BaseModel):
    """Body for POST /action.

    The ``realtime`` and ``fps`` fields are accepted for backwards
    compatibility but are now no-ops: the world free-runs at 60 Hz at all
    times, so action pacing happens automatically.
    """
    actions: List[str]
    realtime: bool = False  # deprecated, no-op
    fps: int = 60           # deprecated, no-op


class SaveRequest(BaseModel):
    """Body for POST /save and POST /load."""
    name: str


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_config: Optional[GameConfig] = None
_runtime: Optional[EmulatorRuntime] = None
_start_time: float = 0.0
_loop: Optional[asyncio.AbstractEventLoop] = None

# WebSocket clients
_ws_clients: Set[WebSocket] = set()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pokemon Agent Server",
    version=__version__,
    description="HTTP + WebSocket API for Pokemon emulator control",
)

# CORS — allow everything for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_game_type(rom_path: str) -> str:
    """Pick reader type based on file extension."""
    ext = Path(rom_path).suffix.lower()
    if ext in (".gb", ".gbc"):
        return "red"
    elif ext == ".gba":
        return "firered"
    raise ValueError(f"Unrecognised ROM extension: {ext}")


def _ensure_runtime() -> EmulatorRuntime:
    """Raise 503 if the runtime isn't ready; return it otherwise."""
    if _runtime is None:
        raise HTTPException(status_code=503, detail="Emulator not initialised")
    return _runtime


async def broadcast(event: dict):
    """Send a JSON event to every connected WebSocket client."""
    dead: List[WebSocket] = []
    payload = json.dumps(event)
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------

# Movement / press timing constants — keep in sync with the legacy semantics
# documented at the previous server.py:217–228.  The agent skill expects:
#   press_X / walk_X : button held for 8 frames, total program duration 20 frames
#   hold_X_N         : button held for N frames, program duration N frames
#   wait_N           : no input, duration N frames
_HOLD_FRAMES = 8
_TOTAL_FRAMES = 20


def _build_program_for_action(action_str: str) -> Program:
    """Parse a single action string into a runtime ``Program``.

    Supported formats:
        press_X            — press button X (held 8 frames, total 20)
        walk_X             — same shape as press, semantically a tile move
        hold_X_N           — hold button X for N frames
        wait_N             — tick N frames with no input
        a_until_dialog_end — press A every 30 frames until dialog clears
    """
    action_str = action_str.strip().lower()

    if action_str == "a_until_dialog_end":
        return AUntilDialogEndProgram()

    parts = action_str.split("_")

    if parts[0] in ("press", "walk") and len(parts) >= 2:
        # walk_up = "_".join(["up"]) etc; press_a = "a"
        button = "_".join(parts[1:])
        return ScheduledProgram(
            edges=[
                InputEdge(0, button, True),
                InputEdge(_HOLD_FRAMES, button, False),
            ],
            duration=_TOTAL_FRAMES,
        )

    if parts[0] == "hold" and len(parts) >= 3:
        button = "_".join(parts[1:-1])
        frames = int(parts[-1])
        return ScheduledProgram(
            edges=[
                InputEdge(0, button, True),
                InputEdge(frames, button, False),
            ],
            duration=frames,
        )

    if parts[0] == "wait" and len(parts) == 2:
        frames = int(parts[1])
        return ScheduledProgram(edges=[], duration=frames)

    raise ValueError(f"Unknown action format: {action_str}")


async def _run_program(program: Program) -> None:
    """Submit a program to the runtime and wait for it to complete."""
    runtime = _ensure_runtime()
    fut = runtime.submit_program(program)
    await asyncio.wrap_future(fut)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def configure(config: GameConfig):
    """Set server configuration (call before app startup)."""
    global _config
    _config = config


@app.on_event("startup")
async def _startup():
    global _runtime, _start_time, _config, _loop
    _loop = asyncio.get_running_loop()
    _start_time = time.time()

    if _config is None:
        # Config can be injected via environment or set beforehand
        print("[server] WARNING: No GameConfig set — emulator will NOT start.")
        print("[server] Call server.configure(GameConfig(...)) before startup.")
        return

    rom = Path(_config.rom_path).expanduser().resolve()
    if not rom.exists():
        print(f"[server] ERROR: ROM not found: {rom}")
        return

    # Auto-detect game type
    game_type = _config.game_type
    if game_type == "auto":
        game_type = _detect_game_type(str(rom))

    print(f"[server] Loading ROM: {rom}")
    print(f"[server] Detected game type: {game_type}")

    # Create emulator
    from pokemon_agent.emulator import create_emulator
    emu = create_emulator(str(rom))

    # Create memory reader
    if game_type == "red":
        from pokemon_agent.memory.red import PokemonRedReader
        reader = PokemonRedReader(emu)
    elif game_type == "firered":
        from pokemon_agent.memory.firered import FireRedMemoryReader
        reader = FireRedMemoryReader(emu)
    else:
        raise ValueError(f"Unknown game type: {game_type}")

    # Create data directories
    data_dir = Path(_config.data_dir).expanduser().resolve()
    (data_dir / "saves").mkdir(parents=True, exist_ok=True)

    # Auto-load a save state if specified — done before runtime starts so the
    # initial published frame reflects the loaded state.
    if _config.load_state:
        saves_dir = data_dir / "saves"
        state_path = saves_dir / f"{_config.load_state}.state"
        if state_path.exists():
            try:
                emu.load_state(str(state_path))
                print(f"[server] Loaded save state: {_config.load_state}")
            except Exception as e:
                print(f"[server] WARNING: Failed to load state '{_config.load_state}': {e}")
        else:
            print(f"[server] WARNING: Save state not found: {state_path}")

    # Spin up the free-running runtime
    _runtime = EmulatorRuntime(emu, reader)
    _runtime.start()

    # Try mounting dashboard
    try:
        import pokemon_agent.dashboard as dashboard_mod  # noqa: F401
        from fastapi.staticfiles import StaticFiles
        dash_dir = Path(dashboard_mod.__file__).parent / "static"
        if dash_dir.is_dir():
            app.mount("/dashboard", StaticFiles(directory=str(dash_dir), html=True), name="dashboard")
            print(f"[server] Dashboard mounted at /dashboard")
        else:
            print("[server] Dashboard module found but no static/ directory")
    except ImportError:
        print("[server] Dashboard not installed — /dashboard unavailable")
        print("[server]   Install with: pip install pokemon-agent[dashboard]")

    print(f"[server] Ready — listening on port {_config.port}")
    print(f"[server] Endpoints:")
    print(f"[server]   GET  /          — server info")
    print(f"[server]   GET  /state     — game state")
    print(f"[server]   GET  /screenshot — current frame (PNG)")
    print(f"[server]   GET  /stream.mjpg — live MJPEG stream")
    print(f"[server]   POST /action    — execute actions")
    print(f"[server]   POST /save      — save state")
    print(f"[server]   POST /load      — load state")
    print(f"[server]   GET  /saves     — list saves")
    print(f"[server]   GET  /minimap   — ASCII minimap")
    print(f"[server]   GET  /health    — health check")
    print(f"[server]   WS   /ws        — live events")


@app.on_event("shutdown")
async def _shutdown():
    """Stop the runtime cleanly so no thread is leaked."""
    global _runtime
    if _runtime is not None:
        _runtime.stop()
        _runtime = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    """Server info."""
    return {
        "name": "pokemon-agent",
        "version": __version__,
        "game": _config.game_type if _config else None,
        "rom": _config.rom_path if _config else None,
        "uptime_seconds": round(time.time() - _start_time, 1) if _start_time else 0,
        "emulator_ready": _runtime is not None,
    }


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "emulator_ready": _runtime is not None}


@app.get("/state")
async def get_state():
    """Full game state JSON (atomic snapshot under the runtime lock)."""
    runtime = _ensure_runtime()
    try:
        state = await asyncio.to_thread(runtime.snapshot_state)
        return JSONResponse(content=state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading state: {e}")


@app.get("/screenshot")
async def screenshot():
    """Current emulator frame as PNG image (from the published latest-frame slot)."""
    runtime = _ensure_runtime()
    try:
        seq, png_bytes = runtime.peek_frame()
        if not png_bytes:
            raise HTTPException(status_code=503, detail="No frame published yet")
        return Response(content=png_bytes, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot error: {e}")


@app.get("/screenshot/base64")
async def screenshot_base64():
    """Current emulator frame as base64-encoded PNG in JSON."""
    runtime = _ensure_runtime()
    try:
        seq, png_bytes = runtime.peek_frame()
        if not png_bytes:
            raise HTTPException(status_code=503, detail="No frame published yet")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return {"image": b64, "format": "png", "seq": seq}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Screenshot error: {e}")


@app.get("/stream.mjpg")
async def mjpeg_stream(fps: int = 15, frames: Optional[int] = None):
    """Live MJPEG stream of the published frame.

    Reads from the runtime's latest-frame slot (no per-request emulator
    access) so the encode cost is shared across all viewers.  ``frames`` is
    primarily for tests; omit for an endless stream.
    """
    runtime = _ensure_runtime()
    fps = max(1, min(int(fps), 30))
    delay = 1.0 / fps
    boundary = "frame"

    async def frame_generator():
        sent = 0
        last_seq = -1
        while frames is None or sent < frames:
            seq, png_bytes = runtime.peek_frame()
            if png_bytes and seq != last_seq:
                last_seq = seq
                yield (
                    b"--" + boundary.encode("ascii") + b"\r\n"
                    b"Content-Type: image/png\r\n"
                    b"Cache-Control: no-cache, no-store, must-revalidate\r\n"
                    b"Pragma: no-cache\r\n"
                    b"Expires: 0\r\n"
                    b"Content-Length: " + str(len(png_bytes)).encode("ascii") + b"\r\n\r\n"
                    + png_bytes
                    + b"\r\n"
                )
                sent += 1
            if frames is None or sent < frames:
                await asyncio.sleep(delay)
        yield b"--" + boundary.encode("ascii") + b"--\r\n"

    return StreamingResponse(
        frame_generator(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.post("/action")
async def execute_actions(req: ActionRequest):
    """Execute a sequence of game actions through the runtime.

    Caller-visible semantics: blocks until all actions complete, returns the
    post-action state. Internally, each action is converted to a frame-
    scheduled program submitted to the runtime thread.

    If ``/action`` is in flight when another request arrives, the second
    program queues behind the first.
    """
    runtime = _ensure_runtime()
    try:
        executed = 0
        for action_str in req.actions:
            program = _build_program_for_action(action_str)
            await _run_program(program)
            executed += 1

        state_after = await asyncio.to_thread(runtime.snapshot_state)

        # Broadcast to WebSocket clients
        await broadcast({
            "type": "action",
            "actions": req.actions,
            "actions_executed": executed,
            "state_after": state_after,
        })

        return {
            "success": True,
            "actions_executed": executed,
            "state_after": state_after,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action error: {e}")


@app.post("/save")
async def save_state(req: SaveRequest):
    """Save emulator state to disk (barriered through the runtime)."""
    runtime = _ensure_runtime()
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    try:
        saves_dir = Path(_config.data_dir).expanduser().resolve() / "saves"
        saves_dir.mkdir(parents=True, exist_ok=True)
        save_path = saves_dir / f"{req.name}.state"
        await asyncio.wrap_future(runtime.save_state(str(save_path)))
        return {"success": True, "path": str(save_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Save error: {e}")


@app.post("/load")
async def load_state(req: SaveRequest):
    """Load emulator state from disk (barriered through the runtime)."""
    runtime = _ensure_runtime()
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    try:
        saves_dir = Path(_config.data_dir).expanduser().resolve() / "saves"
        save_path = saves_dir / f"{req.name}.state"
        if not save_path.exists():
            raise HTTPException(status_code=404, detail=f"Save not found: {req.name}")
        await asyncio.wrap_future(runtime.load_state(str(save_path)))
        state_after = await asyncio.to_thread(runtime.snapshot_state)

        await broadcast({"type": "state_update", "reason": "load", "state": state_after})

        return {"success": True, "name": req.name, "state_after": state_after}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load error: {e}")


@app.get("/saves")
async def list_saves():
    """List available save-state files."""
    if not _config:
        raise HTTPException(status_code=503, detail="Server not configured")
    try:
        saves_dir = Path(_config.data_dir).expanduser().resolve() / "saves"
        if not saves_dir.exists():
            return {"saves": []}
        files = sorted(saves_dir.glob("*.state"))
        saves = [
            {
                "name": f.stem,
                "file": f.name,
                "size_bytes": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
            for f in files
        ]
        return {"saves": saves}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing saves: {e}")


@app.get("/minimap")
async def minimap():
    """Simple ASCII minimap — current map name + player position."""
    runtime = _ensure_runtime()
    try:
        state = await asyncio.to_thread(runtime.snapshot_state)
        map_info = state.get("map", {}) or {}
        player = state.get("player", {}) or {}
        map_name = map_info.get("map_name", "Unknown")
        pos = player.get("position", {}) or {}
        x = pos.get("x", "?")
        y = pos.get("y", "?")

        lines = [
            f"=== {map_name} ===",
            f"Player position: ({x}, {y})",
            "",
            "  N",
            "W + E",
            "  S",
        ]
        text = "\n".join(lines)
        return Response(content=text, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Minimap error: {e}")


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Live event stream via WebSocket."""
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Send a welcome message
        await ws.send_json({
            "type": "connected",
            "version": __version__,
            "emulator_ready": _runtime is not None,
        })
        # Keep alive — wait for client messages (or disconnect)
        while True:
            data = await ws.receive_text()
            # Clients can send a "ping" to keep alive
            if data.strip().lower() == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)


# ---------------------------------------------------------------------------
# Dashboard fallback — only registered if dashboard static files are missing
# ---------------------------------------------------------------------------

def _register_dashboard_fallback():
    """Register a fallback route for /dashboard if static files aren't available."""
    try:
        import pokemon_agent.dashboard as _dm
        static_dir = Path(_dm.__file__).parent / "static"
        if static_dir.is_dir() and (static_dir / "index.html").exists():
            return  # Dashboard exists — don't register fallback
    except ImportError:
        pass

    @app.get("/dashboard")
    @app.get("/dashboard/{path:path}")
    async def dashboard_fallback(path: str = ""):
        raise HTTPException(
            status_code=404,
            detail="Dashboard not installed. Install with: pip install pokemon-agent[dashboard]",
        )

_register_dashboard_fallback()
