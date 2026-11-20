"""Build the 8-bit MotionBricks SOMA model for the browser.

Takes NVIDIA's FP32 ONNX file, quantises the matrix multiplies to 8-bit
weights (activations are quantised on the fly; convolutions stay in float,
since quantising them visibly breaks the motion, per motionbricks-web's
notes), gzips the result and splits it into 19 MB parts that the browser
streams through DecompressionStream. The parts are not named ".gz" so web
hosts do not decompress them on the way.

The model is under the NVIDIA Open Model Agreement, which allows sharing a
modified model; nothing here is committed to git: CI builds it on deploy.

Usage:
    python tools/build_int8.py --out DIR [--fp32 PATH]   (downloads FP32 if no path)
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

FP32_URL = "https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx"
PART_BYTES = 19_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fp32", type=Path)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    fp32 = args.fp32
    if fp32 is None:
        fp32 = out / "mmm_v6_soma_fp32.onnx"
        if not fp32.exists():
            print(f"downloading {FP32_URL}", file=sys.stderr)
            urllib.request.urlretrieve(FP32_URL, fp32)
    fp32_sha = sha256(fp32)
    print(f"fp32 {fp32.stat().st_size / 1e6:.0f} MB sha256 {fp32_sha}", file=sys.stderr)

    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8 = out / "mmm_v6_soma_int8.onnx"
    quantize_dynamic(
        str(fp32), str(int8),
        op_types_to_quantize=["MatMul", "Gemm"],
        weight_type=QuantType.QUInt8,
        extra_options={"MatMulConstBOnly": True},
    )
    raw = int8.read_bytes()
    int8_sha = hashlib.sha256(raw).hexdigest()
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    parts = []
    for i in range(0, len(packed), PART_BYTES):
        name = f"part{i // PART_BYTES:03d}"
        (out / name).write_bytes(packed[i:i + PART_BYTES])
        parts.append(name)
    manifest = {
        "name": "MotionBricks SOMA, 8-bit matrix multiplies (improteca build)",
        "source": FP32_URL,
        "source_sha256": fp32_sha,
        "sha256": int8_sha,
        "bytes": len(raw),
        "packed": len(packed),
        "encoding": "gzip",
        "parts": parts,
        "licence": "NVIDIA Open Model Agreement (modified model)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    int8.unlink()
    if args.fp32 is None:
        fp32.unlink()
    print(f"int8 {len(raw) / 1e6:.0f} MB, {len(packed) / 1e6:.0f} MB gzipped in {len(parts)} parts -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
