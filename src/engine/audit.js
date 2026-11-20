// Physiology and physics checks on SOMA motion: what a constraint layer
// should fix and what a functional visualisation should show.
//
// Per frame (30 Hz), for a list of local poses and optional foot contacts:
//   hinge      knees and elbows are hinges about their local Z axis in SOMA
//              (as in NVIDIA's training data); report sideways rotation
//              (off-axis) and flexion outside 0-150 degrees
//   floor      any toe, heel, hand or the head below the floor
//   slide      a foot the model marks as in contact moving over the floor
//   float      no foot near the floor and no contact, for too long (hovering)
//   jerk       sudden changes of joint velocity

import { TF, qmul, qinv, qToAxisAngle } from './math.js';

export const HINGES = ['LeftShin', 'RightShin', 'LeftForeArm', 'RightForeArm'];
export const FLEX_RANGE = [-0.09, 2.62]; // -5 to 150 degrees about +Z
const FLOOR_PARTS = ['LeftToeBase', 'RightToeBase', 'LeftFoot', 'RightFoot', 'LeftHand', 'RightHand', 'Head'];
// Contact order from the model: left heel, left toe, right heel, right toe.
const CONTACT_JOINTS = ['LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase'];

/** Rotation of a joint relative to its rest local rotation, as a rotation vector (axis * angle). */
export function relRotation(skeleton, local, j, o = 0) {
  const r = skeleton.reference;
  const rest = [r[j * TF + 3], r[j * TF + 4], r[j * TF + 5], r[j * TF + 6]];
  const q = [local[o + j * TF + 3], local[o + j * TF + 4], local[o + j * TF + 5], local[o + j * TF + 6]];
  const { axis, angle } = qToAxisAngle(qmul(qinv(rest), q));
  return [axis[0] * angle, axis[1] * angle, axis[2] * angle];
}

/**
 * @param {import('./skeleton.js').Skeleton} skeleton
 * @param {Float64Array[]} frames  local poses
 * @param {Float32Array[]|null} contacts  per frame, 4 contact probabilities
 */
export function audit(skeleton, frames, contacts = null) {
  const J = skeleton.numJoints;
  const hinge = HINGES.map((n) => skeleton.jointIndex(n));
  const floorParts = FLOOR_PARTS.map((n) => skeleton.jointIndex(n));
  const contactJ = CONTACT_JOINTS.map((n) => skeleton.jointIndex(n));
  const per = [];
  let prevWorld = null;
  let prevVel = null;
  const w = new Float64Array(J * TF);
  for (let f = 0; f < frames.length; f++) {
    const pose = frames[f];
    skeleton.world(pose, w);
    const out = { offAxis: 0, flexOut: 0, below: 0, slide: 0, jerk: 0 };
    for (const j of hinge) {
      const v = relRotation(skeleton, pose, j);
      out.offAxis = Math.max(out.offAxis, Math.hypot(v[0], v[1]));
      const flex = v[2];
      out.flexOut = Math.max(out.flexOut, flex < FLEX_RANGE[0] ? FLEX_RANGE[0] - flex : flex > FLEX_RANGE[1] ? flex - FLEX_RANGE[1] : 0);
    }
    for (const j of floorParts) out.below = Math.max(out.below, -w[j * TF + 1]);
    if (prevWorld && contacts) {
      for (let c = 0; c < 4; c++) {
        if (contacts[f][c] < 0.5) continue;
        const j = contactJ[c] * TF;
        out.slide = Math.max(out.slide, Math.hypot(w[j] - prevWorld[j], w[j + 2] - prevWorld[j + 2]) * 30);
      }
    }
    // Joint velocity change (world positions of hands, feet, head), m/s per frame.
    const vel = floorParts.map((j) => (prevWorld ? [0, 1, 2].map((k) => (w[j * TF + k] - prevWorld[j * TF + k]) * 30) : [0, 0, 0]));
    if (prevVel) out.jerk = Math.max(...vel.map((v, i) => Math.hypot(v[0] - prevVel[i][0], v[1] - prevVel[i][1], v[2] - prevVel[i][2])));
    prevVel = vel;
    prevWorld = Float64Array.from(w);
    per.push(out);
  }
  const deg = 180 / Math.PI;
  const pct = (key, q) => {
    const s = per.map((p) => p[key]).sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor(s.length * q))];
  };
  const frac = (key, threshold) => per.filter((p) => p[key] > threshold).length / Math.max(1, per.length);
  return {
    frames: per,
    summary: {
      offAxisDegP99: pct('offAxis', 0.99) * deg,
      offAxisFrames: frac('offAxis', 10 / deg),
      flexOutFrames: frac('flexOut', 0.02),
      belowFloorFrames: frac('below', 0.02),
      belowFloorMaxCm: pct('below', 1) * 100,
      slideP90: contacts ? pct('slide', 0.9) : null,
      slideFrames: contacts ? frac('slide', 0.15) : null,
      jerkP99: pct('jerk', 0.99),
    },
  };
}
