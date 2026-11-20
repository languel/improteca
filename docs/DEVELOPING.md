# Developing

## Layout

```
src/engine/   the motion engine: a port of NVIDIA's MotionBricks SOMA runtime
  math.js         quaternions and transforms with Warp's conventions
  skeleton.js     joint hierarchy, local -> world
  bvh.js          BVH clips, sampling, remapping onto the SOMA skeleton
  soma.js         encode poses for the model, decode its output, endpoint correction
  inbetweener.js  the generate / play / feed-back loop of NVIDIA's Inbetweener demo
  ort.js          ONNX Runtime adapter (web or node) and the model URLs
assets/soma/  the SOMA skeleton (names, parents, rest pose)
reference/    a script that runs NVIDIA's own pipeline on CPU to make test fixtures
tests/        node --test
local/        model, clips and fixtures for testing (git-ignored)
```

## Tests

```bash
npm install
npm test
```

The math tests always run. The parity and smoothness tests need three local
things, and skip without them:

1. **The model**, `local/models/mmm_v6_soma_fp16.onnx` (380 MB):

   ```bash
   mkdir -p local/models && curl -L -o local/models/mmm_v6_soma_fp16.onnx \
     https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx
   ```

   SHA-256 `6c9f78e23decbda8e1cfd1b0cc96980ea4b24b3ca2c808d03b058f48f82fc9ef`.

2. **NVIDIA's sample clips** in `local/clips/`. They are under NVIDIA's
   evaluation-only sample licence, so they are never committed:
   `idle_loop_001__A268.bvh`, `neutral_looking_around_R_001__A542.bvh`,
   `Neutral_walk_forward_002__A057.bvh`, `jump_over_obstacle_0_5m_R_001__A415.bvh`
   from `assets/motions/soma/` in [NVlabs/motionbricks](https://github.com/NVlabs/motionbricks)
   (Git LFS; the media URL pattern above works for them too).

3. **Reference fixtures**, `local/fixtures/reference.json`, made by running
   NVIDIA's own Warp kernels on the CPU with ONNX Runtime in place of
   TensorRT (no GPU needed; works on a Mac):

   ```bash
   git clone https://github.com/NVlabs/motionbricks ../motionbricks   # then fetch the LFS USD file:
   curl -L -o ../motionbricks/motionbricks/assets/human/soma_base_skel_minimal_no_root.usd \
     https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets/human/soma_base_skel_minimal_no_root.usd
   python3 -m venv .venv && .venv/bin/pip install "warp-lang>=1.17" numpy onnxruntime usd-core scipy
   .venv/bin/python -I reference/make_fixtures.py ../motionbricks local/models/mmm_v6_soma_fp16.onnx local/clips local/fixtures/reference.json
   ```

## What the tests establish

- **Stage by stage, exact.** BVH loading, clip sampling, placed pose windows,
  all seven model inputs, decoding, foot contacts and endpoint correction
  match NVIDIA's kernels (positions within 0.1 mm, rotations within 0.001 rad).
- **End to end, the same behaviour, not bit-exact.** Running the chain with
  onnxruntime-node gives the same token counts, transition points and exact
  endpoints. The motion between endpoints differs slightly (about 1 mm and
  up to 0.11 rad on the first generation) because the model picks discrete
  tokens, and tiny float differences (float64 here, float32 in Warp, and two
  ONNX Runtime builds) can pick a neighbouring one. Differences compound
  over later generations, which start from our own played motion.
- **Handovers.** Over a 12-target, 19-second chain, joints change velocity at
  handovers at most 1.75x as sharply as the motion's own 99th percentile
  (neck; arms 1.5x). These are mild kinks, not pops. An eased endpoint
  correction was tried and did not help, so the kink comes from the model's
  first frames rather than the correction.
- **Speed.** About 40 to 60 ms per generation with ONNX Runtime on CPU (4
  cores), for 1.6 to 1.9 s of motion.
