// Pose transforms for the improviser, on SOMA local poses (joints * 7).

import { TF, getTf, setTf, tf, tp, tq, tmul, tinv, qmul, qinv, qnormalize } from '../engine/math.js';

/** Index of each joint's left/right counterpart (itself for centre joints). */
export function mirrorMap(skeleton) {
  return skeleton.names.map((n) => {
    const other = n.startsWith('Left') ? 'Right' + n.slice(4) : n.startsWith('Right') ? 'Left' + n.slice(5) : n;
    const j = skeleton.jointIndex(other);
    return j < 0 ? skeleton.jointIndex(n) : j;
  });
}

// Reflection across the plane x = 0: positions (x,y,z) -> (-x,y,z), rotations
// (as quaternions) (x,y,z,w) -> (x,-y,-z,w).
const reflectQ = (q) => [q[0], -q[1], -q[2], q[3]];

/**
 * A mirror for one skeleton. SOMA's left and right joint frames are not
 * mirror images of each other, so each joint's reflected world rotation is
 * corrected by the rest frames: R'_j = refl(R_m) * refl(Rest_m)^-1 * Rest_j,
 * where m is j's counterpart.
 */
export function makeMirror(skeleton) {
  const map = mirrorMap(skeleton);
  const J = skeleton.numJoints;
  const restWorld = skeleton.world(skeleton.reference);
  const correction = [];
  for (let j = 0; j < J; j++) {
    const m = map[j];
    const restM = tq(getTf(restWorld, m));
    const restJ = tq(getTf(restWorld, j));
    correction.push(qnormalize(qmul(qinv(reflectQ(restM)), restJ)));
  }
  const world = new Float64Array(J * TF);
  const outWorld = new Float64Array(J * TF);
  /** Mirror one local pose; the root is reflected across x = 0 in the world. */
  return function mirror(local, out = new Float64Array(J * TF)) {
    skeleton.world(local, world);
    for (let j = 0; j < J; j++) {
      const w = getTf(world, map[j]);
      const q = qnormalize(qmul(reflectQ(tq(w)), correction[j]));
      setTf(outWorld, j, tf([-w[0], w[1], w[2]], q));
    }
    // Back to local, keeping the skeleton's own bone offsets.
    setTf(out, 0, getTf(outWorld, 0));
    for (let j = 1; j < J; j++) {
      const parent = getTf(outWorld, skeleton.parents[j]);
      const l = tmul(tinv(parent), getTf(outWorld, j));
      const ref = getTf(skeleton.reference, j);
      setTf(out, j, tf(tp(ref), qnormalize(tq(l))));
    }
    return out;
  };
}

/** Apply a world transform to the root of every pose in a flat array of poses. */
export function placePoses(poses, J, transform) {
  const out = Float64Array.from(poses);
  const n = poses.length / (J * TF);
  for (let p = 0; p < n; p++) setTf(out, p * J, tmul(transform, getTf(poses, p * J)));
  return out;
}
