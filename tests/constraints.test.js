import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { TF, getTf, setTf, tf, tp, tq, qmul, qFromAxisAngle, qnormalize } from '../src/engine/math.js';
import { Constraints, applyJointLimits, JOINT_LIMITS } from '../src/engine/constraints.js';
import { relRotation } from '../src/engine/audit.js';
import { LIMBS, limbInfo, solveLimb, hingeAngle } from '../src/engine/limbs.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const sk = Skeleton.fromJSON(JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8')));
const J = sk.numJoints;
const idx = (n) => sk.jointIndex(n);
const rotate = (pose, name, axis, deg) => {
  const j = idx(name);
  const t = getTf(pose, j);
  setTf(pose, j, tf(tp(t), qnormalize(qmul(tq(t), qFromAxisAngle(axis, (deg * Math.PI) / 180)))));
};
const pos = (w, name) => [w[idx(name) * TF], w[idx(name) * TF + 1], w[idx(name) * TF + 2]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

test('the limb solver keeps hand and foot positions and makes a pure hinge', () => {
  const pose = Float64Array.from(sk.reference);
  rotate(pose, 'LeftArm', [0, 1, 0], 40);
  rotate(pose, 'LeftForeArm', [0.3, 0.6, 0.74], 70); // twisted elbow, as from CMU
  const before = sk.world(pose);
  const world = Float64Array.from(before);
  const info = limbInfo(sk, LIMBS.armL);
  const P = (j) => [world[j * TF], world[j * TF + 1], world[j * TF + 2]];
  const [u, m, e] = [P(info.u), P(info.m), P(info.e)];
  solveLimb(sk, info, pose, world, m.map((x, i) => x - u[i]), e.map((x, i) => x - m[i]));
  const after = sk.world(pose);
  assert.ok(dist(pos(before, 'LeftHand'), pos(after, 'LeftHand')) < 1e-6);
  assert.ok(dist(pos(before, 'LeftForeArm'), pos(after, 'LeftForeArm')) < 1e-6);
  const v = relRotation(sk, pose, idx('LeftForeArm'));
  assert.ok(Math.hypot(v[0], v[1]) < 1e-6, 'elbow is a pure hinge');
  assert.ok(hingeAngle(info, pose) > 0);
});

test('joint limits clamp hyperextension and flipped ankles', () => {
  const pose = Float64Array.from(sk.reference);
  rotate(pose, 'LeftShin', [0, 0, 1], -40); // knee bent backwards
  rotate(pose, 'RightFoot', [1, 0, 0], 170); // flipped ankle
  rotate(pose, 'Head', [0, 1, 0], 120); // head turned too far
  applyJointLimits(sk, pose, 1);
  const knee = limbInfo(sk, LIMBS.legL);
  assert.ok(hingeAngle(knee, pose) >= -5.01 * (Math.PI / 180), 'no hyperextension');
  for (const n of ['RightFoot', 'Head']) {
    const a = Math.hypot(...relRotation(sk, pose, idx(n))) * (180 / Math.PI);
    assert.ok(a <= JOINT_LIMITS[n] + 0.01, `${n} at ${a}°`);
  }
});

test('a planted foot stays put while the body moves; the floor lifts the body', () => {
  const c = new Constraints(sk, { limits: 1, feet: 1, floor: 1, balance: 0 });
  const contact = new Float32Array([1, 1, 0, 0]); // left heel and toe down
  const plantedAt = [];
  for (let f = 0; f < 30; f++) {
    const pose = Float64Array.from(sk.reference);
    // Knees softly bent (a straight leg cannot reach anywhere new).
    rotate(pose, 'LeftLeg', [0, 0, 1], -25);
    rotate(pose, 'LeftShin', [0, 0, 1], 45);
    pose[0] += f * 0.002; // body drifts sideways 6 cm over a second
    pose[1] -= 0.02; // slightly sunk into the floor
    c.frame(pose, contact);
    const w = sk.world(pose);
    if (f >= 5) plantedAt.push(pos(w, 'LeftFoot'));
    const lowest = Math.min(...['LeftToeBase', 'RightToeBase'].map((n) => pos(w, n)[1]));
    assert.ok(lowest >= 0.0149, `toe below the floor: ${lowest}`);
  }
  const drift = Math.max(...plantedAt.map((p) => dist(p, plantedAt[0])));
  assert.ok(drift < 0.01, `planted foot moved ${drift} m`);
});
