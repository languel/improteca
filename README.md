# improteca

An intelligent virtual dancer for the browser. It can be **taught movements**,
**improvises** with what it knows, and can be **guided** by a prompt, by
real-time interaction (a webcam, a partner), or by a choreographic score.

The name is *improvisation* + *-teca* (a collection, as in *biblioteca*): a
library of movement that the dancer draws on and transforms while it dances.

William Forsythe's *Improvisation Technologies* is one inspiration among many:
the idea that improvisation is a set of operations on the body and space
(inscribing lines, transferring the lead to another body part, reorienting,
dropping a point, recombining a phrase), rather than randomness.

## Status

A working demo: **tell the dancer how to dance** ("slow and low, like sinking
into water", "vigorous ballet with jumps and turns"). It reads the prompt as
intentions, improvises with movements it has learned from motion capture in
the [mediathèque](https://languel.github.io/mediatheque/) (CMU ballet,
contemporary and everyday clips, retargeted onto NVIDIA's SOMA body), and
moves with [NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks) between
them, in the browser. Every choice is logged with its reasons and can be
replayed from its seed.

- **Prompt**: a built-in movement vocabulary always works. A language model
  (the **Brain** panel) reads freer language and imagery: LFM3-Omni-2B or
  LFM3-1.7B on WebGPU in the browser (one-time download, nothing leaves your
  machine), Ollama or LM Studio on your machine, or Claude, OpenAI,
  OpenRouter, Gemini, GitHub Models or any OpenAI-compatible API with your
  own key.
- **Repertoire**: clips are split into movements at natural pauses and
  described (level, size, energy, travel, turning, jumps, which part leads).
- **Improviser**: chooses by intention, continuity with the current pose and
  novelty; mirrors, reverses, re-tempos and keeps the dancer on stage.
- **Phrase memory**: the dancer composes in sections. It states motifs (A, B,
  …) and develops them: repeat, mirror, retrograde, fragment, accumulate
  (1; 1 2; 1 2 3), vary, slower, faster, and returns to earlier ones. Low
  variety ("repetitive") develops more; high ("unpredictable") invents more.
  The page shows the form as it unfolds, e.g. `A · A backwards · A mirrored ·
  A varied · B`.
- **Body**: adjustable biology and physics on the generated motion: knees and
  elbows as true hinges within human range, anatomical limits for spine,
  neck, wrists and ankles, feet planted while they bear weight, nothing below
  the floor, optional balance, and smoothed joins between generated pieces.
- **View**: figures (sticks, a mannequin, NVIDIA's SOMA body, drawn lines,
  markers), long trails for hands, feet, head and pelvis, swept limbs, time
  echoes, footprints and markers where the body was corrected. Presets
  include a functional view and a dark drawn view after *Ghostcatching*
  (Bill T. Jones, Paul Kaiser, Shelley Eshkar, 1999).
- **Films**: the dancer can watch a film from the mediathèque (Sandow, Muybridge,
  dancers of 1887...) or your own video, estimate the pose with MediaPipe, fit
  its body to it and learn the movement. One camera: depth is approximate,
  travel is sideways only, and jumps stay on the floor.
- **Inbetweener** (`inbetweener.html`): place poses by hand and play motion
  through them.

Live at <https://languel.github.io/improteca/> once GitHub Pages is enabled.
Architecture and plans: [docs/DESIGN.md](docs/DESIGN.md); implementing Forsythe's
Improvisation Technologies: [docs/IMPROVISATION_TECHNOLOGIES.md](docs/IMPROVISATION_TECHNOLOGIES.md). Tests and how the
engine was checked against NVIDIA's own code: [docs/DEVELOPING.md](docs/DEVELOPING.md).

## Run

No build step. Serve the folder and open it in a desktop browser with WebGPU
(Chrome or Edge, or Safari 26 on a Mac):

```bash
python3 -m http.server 5310
```

Open <http://127.0.0.1:5310/>, write a prompt and press **Dance**. The first run
downloads the motion model and learns about 20 clips from the mediathèque.

**Motion model** (in the page, or `?model=` and `?backend=` in the URL), so the
options can be compared:

| Model | Size | Notes |
| --- | --- | --- |
| `fp16` | 380 MB | NVIDIA's own FP16 file, loaded from NVIDIA's repository |
| `int8` | 210 MB gzipped | 8-bit build made by `tools/build_int8.py` (CI builds it on deploy) |

`auto` (the default) uses FP16 on WebGPU and 8-bit on WebAssembly. The model
runs in a Web Worker, so the page stays smooth while it generates. For the
8-bit build locally: `python tools/build_int8.py --out models/int8`.

## Acknowledgements

The 8-bit model, the Web Worker and streamed gzip parts follow lessons from
[motionbricks-web](https://github.com/eureka-studio-hq/motionbricks-web) by
Eureka Studios.

## Principles

- **Runs in the browser.** WebGPU for the learned motion model, three.js for the
  stage. No server needed for dancing.
- **Learned motion, chosen moves.** A pretrained motion model
  ([NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks)) makes the body
  move like a person between poses. What to dance next is chosen by the
  improviser, from movements the dancer has been taught.
- **Teaching means adding to the repertoire.** Movements come from mocap clips
  and the webcam. The model's weights are not retrained.
- **Every choice is legible.** Each move the dancer makes is recorded with what
  it came from, how it was transformed and why.

## Licences

Code: MIT ([LICENSE](LICENSE)). Third-party models and data keep their own
terms: MotionBricks weights are under the NVIDIA Open Model Agreement, and
NVIDIA's sample BVH clips are for evaluation only, so they are never committed
to this repository.
