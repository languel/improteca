# Design

Status: first draft, November 2026. Milestone 1 (the Inbetweener) is built; see the README.

## 1. What we are building

A virtual dancer that:

1. **can be taught**: show it a movement (a mocap clip, or yourself on a
   webcam) and it adds that movement to its repertoire;
2. **improvises**: it keeps dancing by choosing, combining and transforming
   movements from its repertoire, never just looping clips;
3. **can be guided** through three channels that all speak the same language
   underneath:
   - a **prompt** ("slower, stay low, keep returning to the right elbow");
   - **real-time interaction**: a person on a webcam to mirror, echo, answer or
     avoid, or another virtual dancer;
   - a **score**: choreography or dance notation that fixes some choices and
     leaves others open.

The output is a performance that can be watched live, and also recorded,
replayed and explained.

## 2. Lessons from earlier work

Two earlier projects (`motionbrickhouse`, including its `version0` branch, and
`threegrease`) explored this ground. What we keep:

- **Don't write your own bridge to the model.** `version0` drove MotionBricks
  through a pose converter we wrote ourselves, and got shoulder popping,
  target drift and foot sliding that were never resolved. We never ran NVIDIA's
  own pipeline as a baseline. Here, the first milestone is a faithful port of
  NVIDIA's preparation and endpoint correction, checked against their Python
  code before anything is built on it.
- **Give the model poses it has seen.** MotionBricks was trained on 700 hours of
  studio mocap. Target poses invented from geometry (a stick figure reaching
  "high") may be outside what it knows. Taught movements come from real
  human motion, so they are in-distribution; the improviser transforms them
  gently rather than inventing bodies from nothing.
- **The language before the engine.** `version0` spent most of its effort on
  the engine, and the choreographic layer arrived last. Here the engine is
  milestone 1 because nothing works without it, but it stays small, and the
  repertoire and improviser come straight after.
- **Small scope, one page at a time.** No editor, no panels for every feature.
- **Decisions are data** (from the `motionbrickhouse` restart): every choice the
  dancer makes is recorded with its options and reason, and chance uses a seed.

## 3. Architecture

```
            guidance
  prompt ─────┐
  webcam ─────┼──► intentions ──► IMPROVISER ──► target poses ──► MOTION ENGINE ──► body ──► stage
  score ──────┘                     ▲    │                         (MotionBricks)      │
                                    │    └──► performance log                          │
                               REPERTOIRE ◄── teaching (clips, webcam) ◄───────────────┘
                                                                   (what it actually did)
```

### 3.1 Body

The SOMA skeleton: 30 joints, Y up, meters, 30 frames per second. It is the
skeleton MotionBricks speaks, so everything inside the system uses it.
Rendering starts as a simple mannequin in three.js; skinned characters (VRM,
GLB) come later as a display-only retarget.

### 3.2 Motion engine

MotionBricks SOMA (`mmm_v6_soma`, about 200 million parameters) run in the
browser with ONNX Runtime Web on WebGPU. Each generation takes:

- **context**: the last four poses the dancer actually played;
- **targets**: one to four poses to arrive at;

and returns 0.6 to 1.9 s of motion between them. Published browser ports
report about 120 to 145 ms per generation on an M1 Mac, so generating the next
stretch while the current one plays is feasible.

The engine is a faithful port of NVIDIA's runtime: placing poses into model
space (remove the ground position and heading of the last context pose, face
+X), removing the three boundary frames at each end, endpoint correction, and
feeding the played motion back as the next context. Its only job is to get
from where the dancer is to where the improviser asks. It makes no choices.

### 3.3 Repertoire (what the dancer has been taught)

A **movement** is stored as:

- its key poses (SOMA rotations and root), with timing between them;
- the source it came from (clip, frame range, licence), so nothing is
  anonymous;
- descriptors used for choosing and transforming: which body part leads,
  level (high, middle, low), size in the kinesphere, travel, Laban effort
  estimates (speed, weight, flow), symmetry, and a name if someone gave it
  one.

**Teaching** turns a clip into movements: split it at natural pauses (low joint
speed), keep the key poses, compute descriptors. The person teaching can trim,
name and tag the result. Sources:

- **mocap clips** (BVH): CMU and other openly licensed collections, and the
  user's own captures. They need retargeting to SOMA rotations, which is a
  known hard step (see Unknowns).
- **webcam**: MediaPipe gives joint positions, not rotations. A fitting step
  (inverse kinematics onto SOMA with joint limits) is needed. This comes after
  clips work.

Teaching does not retrain the model. NVIDIA's SOMA release has no training
pipeline, and the repertoire approach keeps every learned movement inspectable
and removable.

### 3.4 Improviser

The improviser decides what to dance next, as a stream of target poses with
timing. Each step:

1. **choose** a movement or fragment from the repertoire, weighted by the
   current intentions, what was just danced (for continuity or contrast) and
   the partner (for response);
2. **transform** it with operations, many after Forsythe and Laban:
   - mirror (left and right), retrograde (backwards), reorient (face another
     direction), relocate on the stage;
   - change level (lower or raise the whole movement), scale its size in the
     kinesphere;
   - **transfer**: let another body part lead the same path (the right-hand
     path done by the elbow or the head);
   - fragment, repeat, accumulate, recombine two movements;
   - change effort: the same shape faster, lighter, more sustained;
3. **check** that the result is still a plausible body (joint limits, balance
   roughly over the feet), and fall back to a milder transform if not;
4. **log** the choice: the options, the selected one, the transforms and the
   reason.

Version 1 is a weighted random walk over the repertoire with a seed. Later
versions can add memory of themes, development over time (state, vary, return)
and phrasing.

### 3.5 Guidance

All three channels compile into **intentions**: weights and constraints the
improviser reads.

| Channel | Examples | How it becomes intentions |
| --- | --- | --- |
| Prompt | "stay low and slow", "keep returning to the spiral you learned" | A language model (for example Claude) maps the text to intention parameters and repertoire tags. It does not generate motion. |
| Real-time interaction | a person on a webcam; another virtual dancer | Pose tracking gives the partner's position, speed and shapes. Response modes: mirror, echo with delay, counter (contrast), follow, keep distance. |
| Score | a written sequence; a notation-inspired timeline | Fixes some choices (this movement, at this time, on this spot) and leaves others to improvisation. |

### 3.6 Performance log

Every played movement, its source, its transforms and the reason, plus every
chance choice with its seed. The log replays a performance exactly and makes
it explainable.

## 4. Milestones

1. **Inbetweener in the browser.** Load SOMA clips, pick key poses, chain
   generations with the ported engine on WebGPU, render a mannequin.
   Done when: port output matches NVIDIA's Python pipeline on the same inputs
   (numeric tests), and chaining NVIDIA's own clips shows no popping.
2. **Repertoire.** Import a clip, split it into movements, compute
   descriptors, browse and name them, play any movement through the engine.
3. **Improviser v1.** Seeded weighted walk, with mirror, retrograde, level and
   reorient. Runs indefinitely without repeating itself obviously. Logged.
4. **Webcam.** Teach by showing (fit SOMA rotations), then live response
   (mirror, echo, counter).
5. **Prompt.** Language to intentions.
6. **Score.** Written sequences and notation-inspired timelines.
7. Richer transforms (transfer, inscription), skinned characters, duets.

## 5. Unknowns and risks

- **Does the port fix the popping?** Milestone 1 answers this. If NVIDIA's own
  pipeline pops on our kinds of targets, the problem is the targets, and the
  improviser has to stay closer to recorded poses.
- **Retargeting to SOMA.** Clips from other skeletons need correct joint
  rotations, not just positions. `version0` found positional retargeting
  insufficient. Options: rotation-space retarget with a calibrated rest pose,
  or an IK fit with SOMA joint limits.
- **Transformed poses may leave the model's comfort zone.** Mirroring and
  reorienting are safe; transfer and scaling may not be. Measure, don't
  assume.
- **Teaching data licences.** NVIDIA's sample clips are evaluation-only: fine
  for testing locally, never committed or published. Teaching material for
  published performances must be openly licensed or the user's own.
- **Browser requirements.** WebGPU on desktop Chrome or Edge, and a model
  download of about 380 MB (FP16) on first use. An 8-bit variant is smaller.
- **Prompting has limits.** A prompt can only steer within what the dancer has
  been taught. Brand-new movement from text would need a text-to-motion model
  (ARDY or Kimodo on a server) as a separate "teacher" that adds clips to the
  repertoire.

## 6. References

- MotionBricks: [code](https://github.com/NVlabs/motionbricks),
  [paper](https://arxiv.org/abs/2604.24833), SOMA model card in that
  repository (`docs/soma_model.md`).
- Browser ports to learn from: [three-motionbricks](https://github.com/takahirox/three-motionbricks)
  (WebGPU, pose studio), [monoradio's ONNX conversion](https://huggingface.co/monoradio/motionbricks-soma-fp16-onnx).
- Related models for later: [ARDY](https://github.com/nv-tlabs/ardy) and
  [Kimodo](https://github.com/nv-tlabs/kimodo) (constraints, text), and
  [ARDY Mini](https://github.com/intsuc/ardy-mini) (text only, in the browser).
- Movement: William Forsythe, *Improvisation Technologies*; Rudolf Laban
  (effort, kinesphere); Trisha Brown (accumulation); Merce Cunningham (chance).
  These are interpretations; no recorded movement or video from these sources
  is used.
