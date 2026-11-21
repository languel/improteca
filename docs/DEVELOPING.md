# Developing

## Layout

```
src/engine/   the motion engine: a port of NVIDIA's MotionBricks SOMA runtime
  math.js         quaternions and transforms with Warp's conventions
  skeleton.js     joint hierarchy, local -> world
  bvh.js          BVH clips, sampling, remapping onto the SOMA skeleton
  soma.js         encode poses for the model, decode its output, endpoint correction
  inbetweener.js  the generate / play / feed-back loop of NVIDIA's Inbetweener demo
  retarget.js     CMU (and other) BVH onto the SOMA skeleton; glitch repair
  limbs.js        two-bone arms and legs with true hinge elbows and knees; IK
  constraints.js  adjustable biology and physics on played frames
  audit.js        measures violations (hinges, ranges, floor, sliding, jerk)
  ort.js          ONNX Runtime adapter (web or node) and the model URLs
src/dancer/   the dancer
  repertoire.js   split clips into movements; descriptors; key moments
  improviser.js   choose, transform and place the next movement; the log
  composer.js     phrase memory: state, develop and return to motifs
  transforms.js   mirror (exact, with SOMA's rest frames) and placement
  intentions.js   prompt -> intentions with a built-in movement vocabulary
  brains.js       prompt -> intentions with a language model: WebGPU (LFM3), local servers or cloud APIs
  fit.js          video pose (MediaPipe landmarks) -> SOMA rotations
app/          the pages: dancer.js (index.html), inbetweener.js, stage, figures.js
              (sticks, mannequin, SOMA body, drawn, markers), viz.js (trails, echoes...),
              engine-client.js + model-worker.js (the model in a Web Worker),
              video.js (MediaPipe pose on a film, then fit.js)
assets/soma/  the SOMA skeleton (names, parents, rest pose)
reference/    a script that runs NVIDIA's own pipeline on CPU to make test fixtures
tools/        build_int8.py: the 8-bit browser model; build_body.py: the SOMA body mesh
              (CI runs both on deploy; nothing from NVIDIA is committed)
models/       built models served with the site (git-ignored)
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

   For the retarget, dancer and CMU smoothness tests, also the mediathèque's CMU
   clips as `local/clips/<id>.bvh` (all `cmu-*` items, from
   `https://languel.github.io/mediatheque/items/<id>/motion.bvh`).

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
- **Retargeting CMU.** Every mapped bone points exactly as in the source on
  every frame (calibrated world-rotation transfer; twist carried over), feet
  on the floor. Chains through retargeted CMU dance hand over a little more
  sharply than NVIDIA's own clips: up to 2.0x the 99th-percentile
  acceleration (left leg; arms 1.7-1.8x) versus 1.75x. Mild, but worth
  improving.
- **Speed.** About 40 to 60 ms per generation with ONNX Runtime on CPU (4
  cores), for 1.6 to 1.9 s of motion.
- **The dancer.** Prompts steer the choices strongly (25-step runs: mean
  energy 0.18 for "calm and still" versus 1.80 for "vigorous, jumping and
  turning"; "ballet" picks ballet 24 times in 25; "no ballet" none). Same seed
  and prompt give the same performance. A 40-step travelling run stays within
  4.5 m of centre. In MotionBricks chains, transitions between movements hand
  over smoothly; the sharpest handover found was inside a turning ballet
  phrase (likely the head spotting), not at a transition.
- **Films.** With ideal landmarks the fitter recovers every limb direction
  exactly (test); the spine is approximated by the straight hip-to-shoulder
  line (about 7°). Light smoothing (±1 frame at 15 fps) costs about 2° on fast
  movement in exchange for less detector jitter.
- **Phrase memory.** Each development operation transforms the motif as named
  (mirrored, reversed with each movement backwards, accumulated as
  1; 1 2; 1 2 3, fragments from the start, one movement varied), checked
  in tests. With "repetitive", at least a third of the steps develop motifs;
  "unpredictable" raises the share of new material. A new prompt starts a new
  motif and keeps the old ones.
- **FP16 versus 8-bit.** The 8-bit build (matrix multiplies only, as in
  motionbricks-web) does not reproduce FP16's motion: on identical inputs its
  joints differ by 8-24° on average and it sometimes picks another segment
  length. It is not worse by the measure that matters: along the same
  improvised dance its raw output misses the target poses by 13.3° against
  FP16's 13.2° (test `precision.test.js`), and it hands over no less
  smoothly. It is 31% smaller and faster: 43 against 64 ms per generation on
  CPU in Node, 270 against 423 ms in the browser on single-threaded
  WebAssembly. Not yet measured on WebGPU.
- **The worker.** With the model on the main thread, 12 s of dancing on
  WebAssembly had 13 frames over 100 ms (up to 500 ms). In the worker: none.
- **Biology and physics.** NVIDIA's SOMA data has elbows and knees as pure
  hinges. Our CMU retarget used to copy CMU's forearm twist into the elbow,
  which put targets outside the model's experience; limbs are now solved as
  two-bone chains with true hinges (twist moves to the wrist), CMU's
  calibration T-pose frame is skipped, and glitch frames (180-degree flips
  from old marker swaps) are repaired. On a 40-second improvised dance
  (`audit.js`), from raw output to output with the default constraints:
  elbows/knees off their hinge in 26% of frames -> 0; over-bent or
  hyperextended 1.9% -> 0; feet sliding while marked in contact 59% of
  contact frames -> 48%; nothing below the floor. Balance is optional (off
  by default): it slightly increases foot sliding.
- **Pops at the end of a move.** Three causes, found and fixed:
  (1) corrections were applied twice at a handover (the new segment already
  starts from the corrected body), so they are re-based to zero there;
  (2) a limb solve hit an exact half turn in `rotationBetweenFrames`, which
  then rotated about an arbitrary axis and swung the arm for two frames
  (17 rad/s; regression test in `retarget.test.js`); (3) the model sometimes
  reverses a limb's direction right at a join. Handover smoothing (a
  velocity offset that fades over 10 frames, up to 0.35 rad per frame) now
  brings the worst handover on the CMU chain from 3.0x (raw) to 1.34x the
  motion's 99th-percentile acceleration; no joint faster than 14 rad/s, and
  position jitter near handovers 7.2 mm/frame² (raw 7.9). Foot planting is
  smoothed (contact held for a few frames, critically damped spring) so a
  planted foot does not snap.

## Text to motion (Kimodo)

Optional. The page uses it when `tools/kimodo_server.py` answers at
`http://127.0.0.1:5320` (the **Text to motion** panel says so; `?kimodo=off`
or the panel's checkbox turns it off; `?kimodo=<url>` points elsewhere).
Without it the director borrows poses from learned clips as before.

On Apple Silicon, through [kimodo-turbo](https://github.com/octaviusp/kimodo-turbo)
(a community MLX port; NVIDIA's own code at a pinned commit, patched to build
on arm64), in the git-ignored `local/vendor/`:

```bash
git clone https://github.com/octaviusp/kimodo-turbo.git local/vendor/kimodo-turbo
cd local/vendor/kimodo-turbo
# what scripts/setup.zsh does, without the Rust character autorig we do not use:
mkdir -p .deps out/weights
git clone https://github.com/nv-tlabs/kimodo.git .deps/kimodo
git -C .deps/kimodo checkout 58e781898b3d7e328a676a75d3e338c45dce3ad9
for p in $PWD/patches/*.patch; do git -C .deps/kimodo apply "$p"; done
curl -fsSL https://raw.githubusercontent.com/DLTcollab/sse2neon/v1.9.1/sse2neon.h \
  -o .deps/kimodo/MotionCorrection/src/cpp/Math/sse2neon.h   # SHA-256 78632498…bc92ed
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python cmake pybind11
PATH="$PWD/.venv/bin:$PATH" uv pip install --python .venv/bin/python ./.deps/kimodo 'torch==2.14.1' \
  'mlx==0.32.3' safetensors tokenizers 'huggingface_hub[hf_xet]==1.33.0'
uv pip install --python .venv/bin/python --no-deps -e .
export HF_HOME=$PWD/out/hf
# the motion model from NVIDIA (1.1 GB, NVIDIA Open Model License):
.venv/bin/hf download nvidia/Kimodo-SOMA-RP-v1 --local-dir out/weights/Kimodo-SOMA-RP-v1
# the Llama 3 text encoder as streamable shards (14 GB, Meta Llama 3 licence):
.venv/bin/hf download Javascript/Llama-3-Kimodo-Turbo --revision bcdf39bf070269e6dbab045334c93cb47a73f033 \
  --local-dir out/weights --include 'llm2vec-llama3-8b/*' --include 'licenses/*' --include LICENSE --include NOTICE.txt
```

The encoder shards above are a community re-upload (no Meta token needed).
To build them from Meta's originals instead, accept the Llama 3 licence on
Hugging Face, log in (`hf auth login`) and run `.venv/bin/kimodo-turbo convert`.

Then, from the repository root:

```bash
local/vendor/kimodo-turbo/.venv/bin/python tools/kimodo_server.py   # --port 5320 --steps 100
```

The server converts Kimodo's output to our rig: Kimodo's 77 SOMA joints are
in the same order as `assets/soma/skeleton.json`, its rotations are relative
to a neutral pose whose joint frames are the world frame, so each joint's
world rotation is Kimodo's global rotation times our rest world rotation
(checked: our forward kinematics lands within 4 cm of Kimodo's joint
positions, the difference between the two rigs' bone lengths).

Measured on an M4 Max (64 GB), standing for 8 s, held through MotionBricks
with the body constraints: toes slide 0.026 m/s and hands move 0.07 m/s,
the same as NVIDIA's own studio idle clip, against 0.069 and 0.17 for the
retargeted CMU idle. A clip takes about 2 to 6 s to generate (100 steps);
a jump needs at least 2.5 s of clip to leave the floor.

