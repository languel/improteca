// Popping check: does a chain of generations play without sudden jumps,
// especially at the handovers between generations (where version0's own
// pipeline popped)? Measures each joint's frame-to-frame rotation change in
// the played motion. Needs the local model and clips (see parity.test.js).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { SomaEngine, SAMPLE_RATE } from '../src/engine/soma.js';
import { sampleWindow, Inbetweener } from '../src/engine/inbetweener.js';
import { placement, TF, qmul, qinv, qToAxisAngle } from '../src/engine/math.js';
import { ortRunner } from '../src/engine/ort.js';
import { retarget } from '../src/engine/retarget.js';
import { Constraints } from '../src/engine/constraints.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const FIXTURE = `${root}local/fixtures/reference.json`;
const MODEL = `${root}local/models/mmm_v6_soma_fp16.onnx`;
const CLIPS = `${root}local/clips/`;
const skip = existsSync(FIXTURE) && existsSync(MODEL) ? false : 'local model or fixtures missing';

/** Rotation-velocity vector (axis * angle, rad per frame) from pose b to pose a. */
function velocity(a, ao, b, bo) {
  const qa = [a[ao + 3], a[ao + 4], a[ao + 5], a[ao + 6]];
  const qb = [b[bo + 3], b[bo + 4], b[bo + 5], b[bo + 6]];
  const { axis, angle } = qToAxisAngle(qmul(qinv(qb), qa));
  return [axis[0] * angle, axis[1] * angle, axis[2] * angle];
}

function angle(a, ao, b, bo) {
  let dot = 0;
  for (let k = 3; k < 7; k++) dot += a[ao + k] * b[bo + k];
  const s = dot < 0 ? -1 : 1;
  let d2 = 0;
  for (let k = 3; k < 7; k++) d2 += (a[ao + k] - s * b[bo + k]) ** 2;
  return 4 * Math.asin(Math.min(1, Math.sqrt(d2) / 2));
}

async function chainCheck(label, loadClips, limit, constraints = false) {
  const fx = JSON.parse(readFileSync(FIXTURE, 'utf8'));
  const skeleton = Skeleton.fromJSON(fx.skeleton);
  const J = skeleton.numJoints;
  const ort = await import('onnxruntime-node');
  const session = await ort.InferenceSession.create(MODEL, { executionProviders: ['cpu'] });
  const engine = new SomaEngine({ skeleton, modelJointNames: fx.model_joint_names, run: ortRunner(ort, session) });
  const clips = loadClips(fx, skeleton);

  // Twelve targets across all four clips, turning and travelling.
  const names = Object.keys(clips);
  const targets = [];
  for (let i = 0; i < 12; i++) {
    const clip = clips[names[i % names.length]];
    const t = (0.7 + 0.37 * i) % Math.max(0.1, clip.duration - 0.5);
    const a = i * 1.1;
    targets.push({ poses: sampleWindow(clip, t, placement(1.5 * Math.cos(a), 1.5 * Math.sin(a), a + 1), 0), numPoses: 1 + (i % 4) });
  }
  const player = new Inbetweener(engine, { constraints: constraints ? new Constraints(skeleton) : null });
  player.reset(sampleWindow(Object.values(clips)[0], 1, placement(0, 0, 0), 3));

  // Record the played frames, marking each handover.
  const played = [];
  const handovers = [];
  for (const target of targets) {
    const out = await player.generate(target);
    const start = played.length ? 1 : 0; // frame 0 repeats the last played pose
    if (played.length) handovers.push(played.length);
    for (let f = start; f <= player.endFrame; f++) played.push(out.poses.slice(f * J * TF, (f + 1) * J * TF));
    player.commit();
  }

  // Per joint: speed (rad/s) and acceleration (change of the rotation-velocity
  // vector between frames). A pop is a sudden change of velocity, so compare
  // the acceleration at each handover with the motion's own 99th percentile.
  const tracked = ['LeftShoulder', 'LeftArm', 'RightShoulder', 'RightArm', 'Chest', 'Neck1', 'LeftLeg', 'RightLeg'];
  const report = [];
  let worstRatio = 0;
  for (const name of tracked) {
    const j = skeleton.jointIndex(name);
    const vel = [];
    for (let f = 1; f < played.length; f++) vel.push(velocity(played[f], j * TF, played[f - 1], j * TF));
    const acc = [];
    for (let f = 1; f < vel.length; f++) acc.push(Math.hypot(vel[f][0] - vel[f - 1][0], vel[f][1] - vel[f - 1][1], vel[f][2] - vel[f - 1][2]));
    // vel[k] is the step into frame k+1; acc[k] compares steps k and k+1.
    // Around handover frame h: acc[h - 2] and acc[h - 1].
    const atHandover = Math.max(...handovers.map((h) => Math.max(acc[h - 2], acc[h - 1])));
    const sorted = [...acc].sort((a, b) => a - b);
    const p99 = sorted[Math.floor(sorted.length * 0.99)];
    const ratio = atHandover / p99;
    worstRatio = Math.max(worstRatio, ratio);
    const speed = Math.max(...vel.map((v) => Math.hypot(...v))) * SAMPLE_RATE;
    report.push(`${name.padEnd(13)} max speed ${speed.toFixed(1)} rad/s; acceleration p99 ${(p99 * SAMPLE_RATE * SAMPLE_RATE).toFixed(0)}, ` +
      `worst handover ${(atHandover * SAMPLE_RATE * SAMPLE_RATE).toFixed(0)} rad/s² (${ratio.toFixed(2)}x)`);
  }
  console.log(`  ${label}: ${played.length} frames (${(played.length / SAMPLE_RATE).toFixed(1)} s), ${handovers.length} handovers`);
  report.forEach((r) => console.log(`  ${r}`));
  // A pop would be a handover acceleration well beyond the motion's own sharp
  // moments. Measured baseline with NVIDIA's pipeline on this chain: up to
  // 1.75x (neck), mild kinks rather than pops. An eased endpoint correction
  // was tried and did not help (legs got worse), so the kink comes from the
  // model's first frames. This guards against regressions.
  assert.ok(worstRatio < limit, `${label}: a handover's acceleration is ${worstRatio.toFixed(2)}x the 99th percentile`);
  // No joint spins (a flip of a hinge or a clamp shows as a burst of speed).
  let fastest = 0;
  for (let j = 1; j < J; j++) {
    for (let f = 1; f < played.length; f++) fastest = Math.max(fastest, angle(played[f], j * TF, played[f - 1], j * TF) * SAMPLE_RATE);
  }
  console.log(`  fastest joint ${fastest.toFixed(1)} rad/s`);
  assert.ok(fastest < 20, `${label}: a joint spins at ${fastest.toFixed(1)} rad/s`);
}

test('a long chain of NVIDIA clips plays without pops at handovers', { skip, timeout: 600_000 }, () =>
  chainCheck('NVIDIA clips', (fx, skeleton) => Object.fromEntries(Object.entries(fx.clips).map(([k, f]) =>
    [k, parseBVH(readFileSync(CLIPS + f, 'utf8'), { skeleton })])), 2));

const CMU = ['cmu-everyday-waiting-idle-139-02', 'cmu-ballet-pirouette-05-02', 'cmu-ballet-arabesque-05-03',
  'cmu-dance-arching-arms-49-12', 'cmu-dance-balance-tilt-49-22', 'cmu-ballet-pointe-tendue-05-12'];
const haveCMU = CMU.every((id) => existsSync(`${CLIPS}${id}.bvh`));
test('a long chain of retargeted CMU dance plays without pops at handovers', { skip: skip || (!haveCMU && 'CMU clips missing'), timeout: 600_000 }, () =>
  chainCheck('CMU dance', (fx, skeleton) => Object.fromEntries(CMU.map((id) =>
    [id, retarget(parseBVH(readFileSync(`${CLIPS}${id}.bvh`, 'utf8')), skeleton)])),
  // With the default constraints, as the app plays it. Measured: up to 1.9x
  // (raw model output on these targets: 3.0x at one right-arm handover).
  2.5, true));
