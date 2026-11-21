"""A local Kimodo server for improteca: text (and optional constraints) in,
SOMA motion out, ready for MotionBricks and the body constraints in the page.

Kimodo (NVIDIA, text-to-motion on the same SOMA skeleton as MotionBricks)
runs here through kimodo-turbo (MLX on Apple Silicon). The page asks for one
motion per directed step; the director places it at the dancer and
MotionBricks in-betweens it live.

Run with kimodo-turbo's environment (see docs/DEVELOPING.md):

    local/vendor/kimodo-turbo/.venv/bin/python tools/kimodo_server.py [--port 5320] [--steps 100]

Only 127.0.0.1 is served. Requests from other origins than local pages and
the published site are refused by the browser (CORS).

GET  /health    {"ok": true, "model": ..., "fps": 30, "joints": 77}
POST /generate  {"text": str, "seconds": 0.5..10, "seed": int?, "steps": int?,
                 "path": {"frames": [int], "xz": [[x, z]], "heading": [radians]?}?}
  -> {"fps": 30, "frames": T, "joints": 77,
      "local": base64 float32 [T, 77, 7] (position xyz, quaternion xyzw; our rig),
      "contacts": [[left heel, left toe, right heel, right toe]] per frame,
      "ms": generation time}

The path is in the motion's own frame: it starts at the origin facing +Z
(heading 0; heading increases toward +X), Y up, metres. Poses come back in
the same frame, as local joint transforms on improteca's SOMA rig
(assets/soma/skeleton.json): Kimodo's rotations are relative to its own
neutral pose, where every joint frame is the world frame, so each joint's
world rotation becomes Kimodo's global rotation times our rig's rest world
rotation, and bone offsets are our rig's.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
MAX_SECONDS = 10.0
ALLOWED_ORIGINS = ("http://localhost:", "http://127.0.0.1:", "https://languel.github.io")


class Rig:
    """improteca's SOMA rig: joint order, parents, rest offsets and rest world rotations."""

    def __init__(self, path: Path):
        s = json.loads(path.read_text())
        self.names = s["joint_names"]
        self.parents = s["parents"]
        ref = np.asarray(s["reference_local"], dtype=np.float64).reshape(len(self.names), 7)
        self.offsets = ref[:, :3]
        rest = [None] * len(self.names)
        for j, p in enumerate(self.parents):
            q = R.from_quat(ref[j, 3:])
            rest[j] = q if p < 0 else rest[p] * q
        self.rest_world = np.stack([r.as_matrix() for r in rest])  # [J, 3, 3]

    def from_kimodo(self, global_rot_mats: np.ndarray, root_positions: np.ndarray) -> np.ndarray:
        """Kimodo global rotations [T, J, 3, 3] and root positions [T, 3] -> local transforms [T, J, 7]."""
        T, J = global_rot_mats.shape[:2]
        world = global_rot_mats @ self.rest_world[None]  # [T, J, 3, 3]
        local_rot = np.empty_like(world)
        for j, p in enumerate(self.parents):
            local_rot[:, j] = world[:, j] if p < 0 else np.swapaxes(world[:, p], -1, -2) @ world[:, j]
        out = np.empty((T, J, 7))
        out[:, :, :3] = self.offsets[None]
        out[:, 0, :3] = root_positions
        out[:, :, 3:] = R.from_matrix(local_rot.reshape(-1, 3, 3)).as_quat().reshape(T, J, 4)
        return out


class Kimodo:
    def __init__(self, steps: int):
        tool = ROOT / "local" / "vendor" / "kimodo-turbo"
        os.environ.setdefault("HF_HOME", str(tool / "out" / "hf"))
        os.chdir(tool)  # kimodo-turbo keeps weights and caches next to itself
        import torch
        from kimodo.constraints import Root2DConstraintSet
        from kimodo_turbo.cli import _turbo_load_model

        self.torch = torch
        self.Root2D = Root2DConstraintSet
        self.steps = steps
        self.model, self.name = _turbo_load_model(return_resolved_name=True)
        self.rig = Rig(ROOT / "assets" / "soma" / "skeleton.json")
        names = [b[0] for b in self.model.skeleton.somaskel77.bone_order_names_with_parents] if hasattr(
            self.model.skeleton, "somaskel77") else None
        if names and names != self.rig.names:
            raise SystemExit("Kimodo's SOMA joint order differs from assets/soma/skeleton.json")
        self.lock = threading.Lock()

    def generate(self, text: str, seconds: float, seed: int | None = None, steps: int | None = None,
                 path: dict | None = None) -> dict:
        torch = self.torch
        frames = max(8, int(round(min(MAX_SECONDS, max(0.3, seconds)) * FPS)))
        constraints = []
        if path and path.get("frames"):
            idx = [min(frames - 1, max(0, int(f))) for f in path["frames"]]
            xz = torch.tensor(path["xz"], dtype=torch.float32)
            heading = None
            if path.get("heading") is not None:
                h = torch.tensor(path["heading"], dtype=torch.float32)
                heading = torch.stack([torch.cos(h), torch.sin(h)], dim=-1)
            constraints.append(self.Root2D(self.model.skeleton, frame_indices=torch.tensor(idx), smooth_root_2d=xz,
                                           global_root_heading=heading))
        with self.lock:
            if seed is not None:
                torch.manual_seed(int(seed))
                np.random.seed(int(seed) % (2**32))
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self.model([text], num_frames=frames, num_denoising_steps=int(steps or self.steps),
                                 constraint_lst=constraints, post_processing=True, return_numpy=True)
            ms = (time.perf_counter() - t0) * 1000
        g = np.asarray(out["global_rot_mats"]).reshape(frames, -1, 3, 3)
        if g.shape[1] != len(self.rig.names):
            raise RuntimeError(f"expected {len(self.rig.names)} joints, got {g.shape[1]}")
        local = self.rig.from_kimodo(g, np.asarray(out["root_positions"]).reshape(frames, 3))
        fc = np.asarray(out.get("foot_contacts", np.zeros((frames, 4)))).reshape(frames, -1)
        if fc.shape[1] == 6:  # [L heel, L toe, L toe end, R heel, R toe, R toe end]
            fc = fc[:, [0, 1, 3, 4]]
        return {
            "fps": FPS,
            "frames": frames,
            "joints": len(self.rig.names),
            "local": base64.b64encode(local.astype("<f4").tobytes()).decode("ascii"),
            "contacts": (fc > 0.5).astype(int).tolist(),
            "ms": round(ms),
        }


def make_handler(kimodo: Kimodo):
    class Handler(BaseHTTPRequestHandler):
        def _cors(self):
            origin = self.headers.get("Origin", "")
            if origin.startswith(ALLOWED_ORIGINS):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                # Chrome's private network access: a public page asking this machine.
                self.send_header("Access-Control-Allow-Private-Network", "true")

        def _json(self, status: int, body: dict):
            data = json.dumps(body).encode()
            self.send_response(status)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):  # noqa: N802
            self.send_response(HTTPStatus.NO_CONTENT)
            self._cors()
            self.end_headers()

        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] == "/health":
                self._json(200, {"ok": True, "model": kimodo.name, "fps": FPS, "joints": len(kimodo.rig.names),
                                 "max_seconds": MAX_SECONDS})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.split("?")[0] != "/generate":
                return self._json(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1_000_000:
                    return self._json(413, {"error": "request too large"})
                req = json.loads(self.rfile.read(length) or b"{}")
                text = str(req.get("text", "")).strip()[:500]
                if not text:
                    return self._json(400, {"error": "text is required"})
                out = kimodo.generate(text, float(req.get("seconds", 4)), req.get("seed"), req.get("steps"),
                                      req.get("path"))
                self._json(200, out)
            except Exception as err:  # noqa: BLE001
                print(f"generate failed: {err!r}", file=sys.stderr, flush=True)
                self._json(500, {"error": str(err)})

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}", flush=True)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=5320)
    ap.add_argument("--steps", type=int, default=100, help="diffusion steps (50 is faster, 100 is Kimodo's default)")
    args = ap.parse_args()
    print("Loading Kimodo…", flush=True)
    kimodo = Kimodo(args.steps)
    t0 = time.perf_counter()
    kimodo.generate("A person stands still.", 1.0, seed=0)
    print(f"Warm-up generation: {time.perf_counter() - t0:.1f} s", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(kimodo))
    print(f"Kimodo ready for improteca: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
