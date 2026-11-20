"""Build the SOMA body mesh for the browser.

Reads the skinned mesh in NVIDIA MotionBricks' SOMA character file
(motionbricks/assets/human/soma_base_skel_minimal_no_root.usd, in NVIDIA's
Apache-2.0 code repository) and writes a compact binary for three.js:
triangles, the four strongest joint weights per vertex (three.js's limit),
and each joint's bind transform. Nothing is committed: CI builds it on deploy.

Usage:
    python tools/build_body.py --out DIR [--usd PATH]   (downloads the USD if no path)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

USD_URL = "https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets/human/soma_base_skel_minimal_no_root.usd"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--usd", type=Path)
    ap.add_argument("--skeleton", type=Path, default=Path(__file__).resolve().parents[1] / "assets/soma/skeleton.json")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    usd = args.usd
    if usd is None:
        usd = out / "soma.usd"
        if not usd.exists():
            print(f"downloading {USD_URL}", file=sys.stderr)
            urllib.request.urlretrieve(USD_URL, usd)

    from pxr import Usd, UsdGeom, UsdSkel

    stage = Usd.Stage.Open(str(usd))
    scale = UsdGeom.GetStageMetersPerUnit(stage)  # centimetres -> metres
    skel_prim = stage.GetPrimAtPath("/OUTPUT/c_skeleton_grp/Hips")
    skel = UsdSkel.Skeleton(skel_prim)
    skel_joints = [str(j).split("/")[-1] for j in skel.GetJointsAttr().Get()]
    bind = [np.array(m, dtype=np.float64) for m in skel.GetBindTransformsAttr().Get()]  # world, row-major (USD: row vectors)

    mesh_prim = next(p for p in stage.Traverse() if p.GetTypeName() == "Mesh")
    mesh = UsdGeom.Mesh(mesh_prim)
    binding = UsdSkel.BindingAPI(mesh_prim)
    mesh_joints = binding.GetJointsAttr().Get()
    mesh_joints = [str(j).split("/")[-1] for j in mesh_joints] if mesh_joints else skel_joints
    geom_bind = np.array(binding.GetGeomBindTransformAttr().Get() or np.eye(4), dtype=np.float64)

    pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
    pts = np.c_[pts, np.ones(len(pts))] @ geom_bind  # row vectors
    pts = pts[:, :3] * scale
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    tris = []
    o = 0
    for c in counts:
        face = idx[o:o + c]
        for k in range(1, c - 1):
            tris.append((face[0], face[k], face[k + 1]))
        o += c
    tris = np.array(tris, dtype=np.uint32)

    ji = np.array(binding.GetJointIndicesPrimvar().Get()).reshape(len(pts), -1)
    jw = np.array(binding.GetJointWeightsPrimvar().Get(), dtype=np.float64).reshape(len(pts), -1)
    order = np.argsort(-jw, axis=1)[:, :4]
    top_i = np.take_along_axis(ji, order, axis=1)
    top_w = np.take_along_axis(jw, order, axis=1)
    top_w /= np.maximum(top_w.sum(axis=1, keepdims=True), 1e-9)

    skeleton = json.loads(args.skeleton.read_text())
    names = skeleton["joint_names"]
    to_ours = np.array([names.index(n) if n in names else -1 for n in mesh_joints])
    if (to_ours < 0).any():
        missing = [mesh_joints[i] for i in np.where(to_ours < 0)[0]]
        raise SystemExit(f"mesh joints missing from the SOMA skeleton: {missing}")
    skin_index = to_ours[top_i].astype(np.uint16)

    # Bind world matrices for our joints (column-major for three.js), metres.
    bind_ours = {}
    for j, name in enumerate(skel_joints):
        if name in names:
            m = bind[j].copy()
            m[3, :3] *= scale
            bind_ours[names.index(name)] = m.T.reshape(-1, order="F").tolist()  # USD row-vector -> column-major

    blobs = [
        ("position", pts.astype(np.float32)),
        ("index", tris.reshape(-1)),
        ("skinIndex", skin_index.reshape(-1)),
        ("skinWeight", top_w.astype(np.float32).reshape(-1)),
    ]
    offset = 0
    layout = {}
    with (out / "soma_body.bin").open("wb") as f:
        for name, arr in blobs:
            data = np.ascontiguousarray(arr).tobytes()
            pad = (-offset) % 4
            f.write(b"\0" * pad)
            offset += pad
            layout[name] = {"offset": offset, "length": int(arr.size), "type": str(arr.dtype)}
            f.write(data)
            offset += len(data)
    meta = {
        "source": USD_URL,
        "vertices": int(len(pts)),
        "triangles": int(len(tris)),
        "layout": layout,
        "bind": {str(k): v for k, v in bind_ours.items()},
        "note": "SOMA body from NVIDIA MotionBricks (Apache-2.0 code repository); four strongest joint weights per vertex.",
    }
    (out / "soma_body.json").write_text(json.dumps(meta))
    if args.usd is None:
        usd.unlink()
    print(f"body: {len(pts)} vertices, {len(tris)} triangles, {offset / 1e6:.1f} MB -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
