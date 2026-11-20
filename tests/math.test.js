import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  qmul, qinv, qnormalize, qrot, qFromAxisAngle, qslerp, qFromCols, tmul, tinv, tblend,
  deltaTransform, applyDelta, IDENTITY,
} from '../src/engine/math.js';

const close = (a, b, eps = 1e-9) => a.forEach((x, i) => assert.ok(Math.abs(x - b[i]) < eps, `${a} vs ${b}`));

test('rotation about Y by 90 degrees maps +X to -Z', () => {
  close(qrot(qFromAxisAngle([0, 1, 0], Math.PI / 2), [1, 0, 0]), [0, 0, -1]);
});

test('quaternion times its inverse is identity', () => {
  const q = qnormalize([0.3, -0.2, 0.5, 0.7]);
  close(qmul(q, qinv(q)), [0, 0, 0, 1]);
});

test('slerp endpoints and shortest path', () => {
  const a = qFromAxisAngle([1, 0, 0], 0.4);
  const b = qFromAxisAngle([0, 0, 1], -1.1);
  close(qslerp(a, b, 0), a);
  close(qslerp(a, b, 1), b);
  // Shortest path: slerp to -b equals slerp to b.
  close(qslerp(a, [-b[0], -b[1], -b[2], -b[3]], 0.5), qslerp(a, b, 0.5));
});

test('matrix columns round trip to the same rotation', () => {
  for (const q of [qnormalize([0.1, 0.9, -0.3, 0.2]), qnormalize([0.7, 0.1, 0.1, -0.7]), [0, 0, 0, 1]]) {
    const r = qFromCols(qrot(q, [1, 0, 0]), qrot(q, [0, 1, 0]), qrot(q, [0, 0, 1]));
    const d = Math.abs(r[0] * q[0] + r[1] * q[1] + r[2] * q[2] + r[3] * q[3]);
    assert.ok(Math.abs(d - 1) < 1e-9);
  }
});

test('transform inverse and deltas', () => {
  const a = [1, 2, 3, ...qnormalize([0.2, 0.4, -0.1, 0.9])];
  const b = [-0.5, 0.1, 2, ...qnormalize([-0.3, 0.2, 0.6, 0.5])];
  close(tmul(a, tinv(a)), IDENTITY);
  close(applyDelta(a, deltaTransform(a, b), 1), b);
  close(applyDelta(a, deltaTransform(a, b), 0), a);
  close(tblend(a, b, 1), b);
});
