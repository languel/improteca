// Two-bone limbs (arm: Arm-ForeArm-Hand, leg: Leg-Shin-Foot) with the middle
// joint as a true hinge, as in NVIDIA's SOMA data: the elbow and knee rotate
// only about their local Z axis, between about -5 and 150 degrees.
//
// solveLimb() poses a limb so the middle joint and the end land where asked
// (given as world directions of the two bones, or an end position for IK),
// by turning the upper bone (swing and twist, so the hinge axis is normal to
// the limb's plane) and setting the hinge angle. The end's world rotation is
// kept, so forearm twist moves into the wrist.

import { TF, getTf, setTf, tf, tp, tq, qmul, qinv, qrot, qnormalize, qFromAxisAngle, cross } from './math.js';
import { rotationBetweenFrames, rotationBetween } from './retarget.js';

export const LIMBS = {
  armL: ['LeftArm', 'LeftForeArm', 'LeftHand'],
  armR: ['RightArm', 'RightForeArm', 'RightHand'],
  legL: ['LeftLeg', 'LeftShin', 'LeftFoot'],
  legR: ['RightLeg', 'RightShin', 'RightFoot'],
};
export const HINGE_RANGE = [-5 * (Math.PI / 180), 150 * (Math.PI / 180)];

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => {
  const l = len(a);
  return l > 1e-9 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
};

/** Constant data for one limb on one skeleton. */
export function limbInfo(skeleton, names) {
  const [u, m, e] = names.map((n) => skeleton.jointIndex(n));
  const ref = skeleton.reference;
  const restM = tq(getTf(ref, m));
  const boneU = tp(getTf(ref, m)); // middle joint offset in the upper joint's frame
  const boneM = tp(getTf(ref, e)); // end offset in the middle joint's frame
  return {
    u, m, e,
    restM,
    lenU: len(boneU),
    lenM: len(boneM),
    dirU: norm(boneU), // upper bone direction, upper frame
    dirM: norm(boneM), // lower bone direction, middle frame
    hingeInU: qrot(restM, [0, 0, 1]), // hinge axis in the upper frame
  };
}

/** Signed hinge angle of the middle joint (rotation about local Z relative to rest). */
export function hingeAngle(info, local) {
  const q = qmul(qinv(info.restM), tq(getTf(local, info.m)));
  const s = q[3] < 0 ? -1 : 1;
  return 2 * Math.atan2(s * q[2], s * q[3]);
}

/**
 * Pose a limb in place on `local` (and `world`, which must hold its world
 * transforms and is updated for the limb's three joints).
 * dU, dM: world directions wanted for the upper and lower bones.
 * Options: range [min, max] for the hinge; prevNormal to resolve a straight limb.
 * Returns { angle, normal, clamped }.
 */
export function solveLimb(skeleton, info, local, world, dU, dM, { range = HINGE_RANGE, prevNormal = null } = {}) {
  const { u, m, e } = info;
  const parentQ = tq(getTf(world, skeleton.parents[u]));
  const endWorldQ = tq(getTf(world, e));
  dU = norm(dU);
  dM = norm(dM);
  // The limb's plane normal. For a nearly straight limb the plane is mostly
  // noise, and a hinge would make the upper bone's twist follow it (spinning
  // the arm or leg); blend toward the previous frame's hinge axis (or the
  // input pose's) below about 7 degrees of bend.
  const ref0 = prevNormal || qrot(tq(getTf(world, u)), info.hingeInU);
  let refP = sub(ref0, dU.map((x) => x * dot(ref0, dU)));
  refP = len(refP) > 1e-6 ? norm(refP) : norm(cross(dU, Math.abs(dU[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
  const c = cross(dU, dM);
  const sinBend = len(c);
  let n = refP;
  if (sinBend > 1e-9) {
    let nc = norm(c);
    if (dot(nc, refP) < 0) nc = nc.map((x) => -x);
    const w = Math.min(1, Math.max(0, (sinBend - 0.02) / 0.1));
    n = norm(nc.map((x, i) => x * w + refP[i] * (1 - w)));
  }
  // Two candidate orientations (hinge axis along +n or -n). Prefer one whose
  // angle is in range; when both are (a nearly straight limb), keep the
  // hinge axis closest to the previous frame's, or to the input pose's own,
  // so the upper bone never flips its twist from one frame to the next.
  const current = qrot(tq(getTf(world, u)), info.hingeInU);
  const reference = prevNormal || current;
  let best = null;
  for (const sign of [1, -1]) {
    const axis = n.map((x) => x * sign);
    const Ru = rotationBetweenFrames(info.dirU, info.hingeInU, dU, axis);
    // Lower bone direction at hinge angle 0, then the angle that reaches dM about the axis.
    const c0 = qrot(qmul(Ru, info.restM), info.dirM);
    const angle = Math.atan2(dot(cross(c0, dM), axis), dot(c0, dM));
    const out = angle < range[0] ? range[0] - angle : angle > range[1] ? angle - range[1] : 0;
    // Continuity first: flipping the upper bone 180 degrees to avoid a small
    // range violation looks far worse than clamping, so a flip needs a large one.
    const cost = (1 - dot(axis, reference)) + 2 * out;
    if (!best || cost < best.cost) best = { Ru, angle, out, axis, cost };
  }
  const angle = Math.min(range[1], Math.max(range[0], best.angle));
  const localU = qnormalize(qmul(qinv(parentQ), best.Ru));
  const localM = qnormalize(qmul(info.restM, qFromAxisAngle([0, 0, 1], angle)));
  const lu = getTf(local, u);
  const lm = getTf(local, m);
  setTf(local, u, tf(tp(lu), localU));
  setTf(local, m, tf(tp(lm), localM));
  // World transforms of the limb, then the end keeps its world rotation.
  const pw = getTf(world, skeleton.parents[u]);
  const wu = [...pw.slice(0, 3)];
  const pu = qrot(parentQ, tp(lu));
  const posU = [wu[0] + pu[0], wu[1] + pu[1], wu[2] + pu[2]];
  const qU = best.Ru;
  const posM = (() => {
    const o = qrot(qU, tp(lm));
    return [posU[0] + o[0], posU[1] + o[1], posU[2] + o[2]];
  })();
  const qM = qnormalize(qmul(qU, localM));
  const le = getTf(local, e);
  const posE = (() => {
    const o = qrot(qM, tp(le));
    return [posM[0] + o[0], posM[1] + o[1], posM[2] + o[2]];
  })();
  setTf(world, u, tf(posU, qU));
  setTf(world, m, tf(posM, qM));
  setTf(local, e, tf(tp(le), qnormalize(qmul(qinv(qM), endWorldQ))));
  setTf(world, e, tf(posE, endWorldQ));
  return { angle, normal: best.axis, clamped: best.out > 1e-6, outBy: best.out };
}

/**
 * Two-bone IK: directions that put the end at `target` (world), keeping the
 * middle joint on the side of `pole` (a world point or direction hint).
 * Returns { dU, dM, reach } where reach < 1 means the target was out of reach.
 */
export function ikDirections(info, root, target, poleDir) {
  const a = info.lenU;
  const b = info.lenM;
  let toT = sub(target, root);
  let d = len(toT);
  const maxD = (a + b) * 0.9995;
  const reach = d > maxD ? maxD / d : 1;
  d = Math.min(Math.max(d, Math.abs(a - b) + 1e-4), maxD);
  const axis = norm(toT);
  // Angle at the root between the target line and the upper bone.
  const cosA = (a * a + d * d - b * b) / (2 * a * d);
  const angA = Math.acos(Math.min(1, Math.max(-1, cosA)));
  // Bend toward the pole, perpendicular to the target line.
  let side = sub(poleDir, axis.map((x) => x * dot(poleDir, axis)));
  if (len(side) < 1e-6) side = Math.abs(axis[1]) < 0.9 ? cross(axis, [0, 1, 0]) : cross(axis, [1, 0, 0]);
  side = norm(side);
  const dU = norm(axis.map((x, i) => x * Math.cos(angA) + side[i] * Math.sin(angA)));
  const mid = [root[0] + dU[0] * a, root[1] + dU[1] * a, root[2] + dU[2] * a];
  const end = [root[0] + axis[0] * d, root[1] + axis[1] * d, root[2] + axis[2] * d];
  return { dU, dM: norm(sub(end, mid)), reach };
}

/** Rotation that turns world direction a onto b (exported for callers). */
export { rotationBetween };
