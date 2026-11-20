// CMU -> SOMA retargeting: every mapped bone points the same way as the
// source bone on every frame, and the feet reach the floor. Uses a CMU clip
// from the mediathèque in local/clips/ (skips without it; see DEVELOPING.md).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { retarget, CMU_TO_SOMA, bvhToSoma, leadingCalibrationFrame, rotationBetweenFrames } from '../src/engine/retarget.js';
import { TF, qrot } from '../src/engine/math.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const CLIP = `${root}local/clips/cmu-ballet-pirouette-05-02.bvh`;
const skip = existsSync(CLIP) ? false : 'local CMU clip missing';
const soma = Skeleton.fromJSON(JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8')));

test('CMU bones keep their directions on SOMA', { skip }, () => {
  const src = parseBVH(readFileSync(CLIP, 'utf8'));
  const out = retarget(src, soma);
  const J = soma.numJoints;
  const bones = [['LeftArm', 'LeftForeArm'], ['LeftForeArm', 'LeftHand'], ['RightArm', 'RightForeArm'],
    ['RightForeArm', 'RightHand'], ['LeftLeg', 'LeftShin'], ['LeftShin', 'LeftFoot'], ['RightLeg', 'RightShin'],
    ['RightShin', 'RightFoot'], ['LeftFoot', 'LeftToeBase'], ['Spine2', 'Chest'], ['Neck2', 'Head']];
  const vec = (w, sk, a, b) => {
    const i = sk.jointIndex(a) * TF;
    const k = sk.jointIndex(b) * TF;
    return [w[k] - w[i], w[k + 1] - w[i + 1], w[k + 2] - w[i + 2]];
  };
  // CMU exports start with a calibration T-pose, which the retarget skips.
  const start = leadingCalibrationFrame(src) ? 1 / src.sampleRate : 0;
  assert.ok(start > 0, 'the calibration frame is detected');
  let worst = 0;
  const errors = [];
  let lowest = Infinity;
  for (let f = 0; f < out.numFrames; f++) {
    const sw = soma.world(out.poses.subarray(f * J * TF, (f + 1) * J * TF));
    const cw = src.skeleton.world(src.sample(start + f / out.sampleRate));
    for (const [a, b] of bones) {
      const s = vec(sw, soma, a, b);
      const c = vec(cw, src.skeleton, CMU_TO_SOMA[a], CMU_TO_SOMA[b]);
      const cos = (s[0] * c[0] + s[1] * c[1] + s[2] * c[2]) / Math.hypot(...s) / Math.hypot(...c);
      const e = Math.acos(Math.min(1, Math.max(-1, cos)));
      errors.push(e);
      worst = Math.max(worst, e);
    }
    for (const n of ['LeftToeBase', 'RightToeBase']) lowest = Math.min(lowest, sw[soma.jointIndex(n) * TF + 1]);
  }
  errors.sort((x, y) => x - y);
  const median = errors[errors.length >> 1];
  // Exact up to the corrections made on purpose: glitch repair, joint limits,
  // and steady bend planes for nearly straight limbs.
  assert.ok(worst < 0.15, `bone direction error ${worst} rad`);
  assert.ok(median < 1e-3, `median bone direction error ${median} rad`);
  assert.ok(Math.abs(lowest - 0.022) < 0.005, `lowest toe at ${lowest} m`);
  assert.equal(out.sampleRate, 30);
});

test('BVH files are recognised as CMU or SOMA', { skip }, () => {
  assert.equal(bvhToSoma(readFileSync(CLIP, 'utf8'), soma, parseBVH).method, 'CMU retarget');
});

test('a frame rotation keeps the primary axis when the secondary turns half way round', () => {
  // The case that swung an arm for two frames: secondaries exactly opposite
  // after aligning the primaries.
  const n = [0, 1, 0];
  const q = rotationBetweenFrames([0, 1, 0], [1, 0, 0], n, [-1, 0, 0]);
  const p = qrot(q, [0, 1, 0]);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(p[i] - n[i]) < 1e-6, `primary moved: ${p} vs ${n}`);
});
