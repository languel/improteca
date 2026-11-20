// Parity with NVIDIA's MotionBricks runtime.
//
// local/fixtures/reference.json is made by reference/make_fixtures.py, which
// runs NVIDIA's own Warp kernels on the CPU with ONNX Runtime. It is derived
// from evaluation-only sample clips, so it is not committed; these tests skip
// when it (or the local model and clips) are missing. See docs/DEVELOPING.md.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { SomaEngine } from '../src/engine/soma.js';
import { sampleWindow, Inbetweener } from '../src/engine/inbetweener.js';
import { placement, TF } from '../src/engine/math.js';
import { ortRunner } from '../src/engine/ort.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const FIXTURE = `${root}local/fixtures/reference.json`;
const MODEL = `${root}local/models/mmm_v6_soma_fp16.onnx`;
const CLIPS = `${root}local/clips/`;
const have = existsSync(FIXTURE) && existsSync(MODEL);
const skip = have ? false : 'local reference fixtures or model missing';

const fx = have ? JSON.parse(readFileSync(FIXTURE, 'utf8')) : null;
const skeleton = have ? Skeleton.fromJSON(fx.skeleton) : null;

// Positions within 1e-4 m (0.1 mm); rotations within an angle of 1e-3 rad.
function assertPoses(actual, expected, label, { pos = 1e-4, ang = 1e-3 } = {}) {
  const exp = expected.flat();
  assert.equal(actual.length, exp.length, `${label}: length`);
  let worstPos = 0;
  let worstAng = 0;
  for (let i = 0; i < actual.length; i += TF) {
    for (let k = 0; k < 3; k++) worstPos = Math.max(worstPos, Math.abs(actual[i + k] - exp[i + k]));
    // Angle between rotations from the quaternion difference (q and -q are the
    // same rotation); unlike acos of a dot product, this stays exact near zero.
    const dot = actual[i + 3] * exp[i + 3] + actual[i + 4] * exp[i + 4] + actual[i + 5] * exp[i + 5] + actual[i + 6] * exp[i + 6];
    const s = dot < 0 ? -1 : 1;
    let d2 = 0;
    for (let k = 3; k < 7; k++) d2 += (actual[i + k] - s * exp[i + k]) ** 2;
    worstAng = Math.max(worstAng, 4 * Math.asin(Math.min(1, Math.sqrt(d2) / 2)));
  }
  assert.ok(worstPos < pos, `${label}: position error ${worstPos}`);
  assert.ok(worstAng < ang, `${label}: rotation error ${worstAng}`);
  return { worstPos, worstAng };
}

function assertArray(actual, expected, label, eps) {
  assert.equal(actual.length, expected.length, `${label}: length`);
  let worst = 0;
  for (let i = 0; i < actual.length; i++) worst = Math.max(worst, Math.abs(actual[i] - expected[i]));
  assert.ok(worst < eps, `${label}: max error ${worst}`);
  return worst;
}

let clipsCache = null;
function clips() {
  if (!clipsCache) {
    clipsCache = {};
    for (const [name, file] of Object.entries(fx.clips)) {
      clipsCache[name] = parseBVH(readFileSync(CLIPS + file, 'utf8'), { skeleton });
    }
  }
  return clipsCache;
}

const tokenOf = (poses) => ({ poses: Float64Array.from(poses.flat()), numPoses: 4 });

test('BVH loading and clip sampling match the reference', { skip }, () => {
  for (const [name, s] of Object.entries(fx.samples)) {
    const clip = clips()[name];
    assert.equal(clip.numFrames, s.num_frames);
    assert.equal(clip.sampleRate, s.sample_rate);
    s.times.forEach((t, i) => assertPoses(clip.sample(t), s.poses[i], `${name} @ ${t}s`));
  }
});

test('placed pose windows match the reference', { skip }, () => {
  fx.plan.forEach((e, i) => {
    const w = sampleWindow(clips()[e.clip], e.time, placement(e.x, e.z, (e.yaw * Math.PI) / 180), i === 0 ? 3 : 0);
    assertPoses(w, fx.windows[i], `window ${i}`);
  });
});

test('encode, decode and endpoint correction match the reference', { skip }, () => {
  const engine = new SomaEngine({ skeleton, modelJointNames: fx.model_joint_names, run: null });
  fx.generations.forEach((g, i) => {
    const entry = fx.plan[i + 1];
    const context = tokenOf(g.context);
    const target = { poses: Float64Array.from(g.target.flat()), numPoses: entry.num_poses };
    const tokens = [];
    for (let c = entry.tokens[0]; c <= entry.tokens[1]; c++) tokens.push(c);
    const { inputs, alignment } = engine.encode(context, target, { allowedTokenCounts: tokens });
    for (const [name, values] of Object.entries(g.inputs)) {
      assertArray(Array.from(inputs[name]), values, `generation ${i + 1} input ${name}`, 2e-4);
    }
    assertArray(alignment, g.alignment, `generation ${i + 1} alignment`, 1e-5);
    // Decode the reference's own model output, so this checks decoding alone.
    const out = engine.decode(Float32Array.from(g.pred_global_motions), g.pred_num_tokens, g.alignment);
    assert.equal(out.frameCount, g.frame_count);
    assertPoses(out.poses, g.decoded, `generation ${i + 1} decoded`);
    assertArray(Array.from(out.contacts), g.contacts.flat(), `generation ${i + 1} contacts`, 1e-6);
    engine.correctEndpoints(out, context, target);
    assertPoses(out.poses, g.corrected, `generation ${i + 1} corrected`);
  });
});

test('the full chain with ONNX Runtime behaves like NVIDIA end to end', { skip, timeout: 600_000 }, async () => {
  const ort = await import('onnxruntime-node');
  const session = await ort.InferenceSession.create(MODEL, { executionProviders: ['cpu'] });
  const engine = new SomaEngine({ skeleton, modelJointNames: fx.model_joint_names, run: ortRunner(ort, session) });
  const player = new Inbetweener(engine);
  const windows = fx.plan.map((e, i) =>
    sampleWindow(clips()[e.clip], e.time, placement(e.x, e.z, (e.yaw * Math.PI) / 180), i === 0 ? 3 : 0));
  const J = skeleton.numJoints;
  const P = J * TF;
  player.reset(windows[0]);
  for (let i = 1; i < fx.plan.length; i++) {
    const e = fx.plan[i];
    const g = fx.generations[i - 1];
    const tokens = [];
    for (let c = e.tokens[0]; c <= e.tokens[1]; c++) tokens.push(c);
    const context = player.context;
    const out = await player.generate({ poses: windows[i], numPoses: e.num_poses, tokens, early: e.early });
    const n = out.frameCount;

    // Deterministic behaviour: same length, same transition point, and endpoint
    // correction lands exactly on our last context pose and first target pose.
    assert.equal(out.tokens, g.pred_num_tokens, `generation ${i} token count`);
    assert.equal(player.endFrame + 1, g.played_frames, `generation ${i} played frames`);
    assertPoses(out.poses.slice(0, P), [Array.from(context.poses.slice(3 * P, 4 * P))], `generation ${i} start`);
    assertPoses(out.poses.slice((n - 1) * P, n * P), [Array.from(windows[i].slice(0, P))], `generation ${i} end`);

    // The motion itself is not bit-exact, by design; the stage-by-stage test
    // above is where exactness is checked. Our float64 encoder (Warp's is
    // float32) and onnxruntime-node (which already differs from Python's
    // onnxruntime by ~2e-4 on identical inputs) feed a model that picks
    // discrete motion tokens, so a tiny input difference can choose a
    // neighbouring token: a slightly different, equally valid in-between.
    // Measured on generation 1 (same inputs to 3e-7): ~1 mm and up to 0.11 rad
    // mid-motion, zero at the ends. Later generations start from our own
    // played motion, so differences compound; they get a sanity bound only.
    const bound = i === 1 ? { pos: 1e-2, ang: 0.2 } : { pos: 0.2, ang: 0.6 };
    const err = assertPoses(out.poses, g.corrected, `generation ${i} motion`, bound);
    player.commit();
    console.log(`  generation ${i}: ${out.tokens} tokens, ${n} frames, ${out.inferenceMs.toFixed(0)} ms on CPU, ` +
      `difference from NVIDIA ${(err.worstPos * 1000).toFixed(1)} mm / ${err.worstAng.toFixed(3)} rad`);
  }
});
