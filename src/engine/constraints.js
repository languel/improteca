// Adjustable biology and physics for generated motion.
//
// MotionBricks makes plausible motion, but it builds each joint's rotation
// independently and endpoint correction drags the body toward its targets,
// so elbows twist, joints over-bend, feet slide and parts can dip below the
// floor. This layer corrects each played frame, in order, with a strength
// per constraint (0 = off, 1 = full):
//
//   limits    knees and elbows as hinges (-5 to 150 degrees), and anatomical
//             ranges for the spine, neck, head, collarbones, wrists, ankles,
//             toes and hips
//   feet      a foot the model marks as in contact stays planted (leg IK),
//             easing in and out of each contact
//   floor     nothing goes below the floor (the body is lifted)
//   balance   the centre of mass is kept over the supporting feet (the hips
//             shift; planted feet stay)
//   smooth    at each handover between generated segments, the first frames
//             blend from the previous motion continued at its own velocity,
//             so velocity changes over about 0.13 s instead of in one frame
//
// The corrected frames are what plays and what the next generation starts
// from, so the model continues from the corrected body.

import { TF, getTf, setTf, tf, tp, tq, qmul, qinv, qnormalize, qToAxisAngle, qFromAxisAngle } from './math.js';
import { LIMBS, limbInfo, solveLimb, ikDirections, hingeAngle, HINGE_RANGE } from './limbs.js';

const DEG = Math.PI / 180;
/** Largest rotation from rest (degrees) for joints without a hinge. */
export const JOINT_LIMITS = {
  Spine1: 45, Spine2: 35, Chest: 30, Neck1: 50, Neck2: 50, Head: 45,
  LeftShoulder: 40, RightShoulder: 40, LeftHand: 90, RightHand: 90,
  LeftLeg: 140, RightLeg: 140, LeftFoot: 70, RightFoot: 70, LeftToeBase: 70, RightToeBase: 70,
};
// Lowest allowed height (m) of a joint's centre: its "radius" above the floor.
const FLOOR_CLEARANCE = { LeftToeBase: 0.015, RightToeBase: 0.015, LeftFoot: 0.05, RightFoot: 0.05, LeftHand: 0.03, RightHand: 0.03, Head: 0.09, Hips: 0.12 };
// Rough segment masses for the centre of mass (fractions of body mass, at joints).
const MASS = { Hips: 0.14, Spine2: 0.16, Chest: 0.2, Head: 0.08, LeftArm: 0.03, RightArm: 0.03, LeftForeArm: 0.02, RightForeArm: 0.02, LeftHand: 0.01, RightHand: 0.01, LeftLeg: 0.1, RightLeg: 0.1, LeftShin: 0.045, RightShin: 0.045, LeftFoot: 0.015, RightFoot: 0.015 };

const HANDOVER_FRAMES = 10;

/** Per joint: rotation vector from pose a to pose b (b relative to a, local), and root velocity. */
function velocity(a, b) {
  const J = a.length / TF;
  const v = new Float64Array(J * 3 + 3);
  for (let j = 0; j < J; j++) {
    const o = j * TF;
    const { axis, angle } = qToAxisAngle(qmul(qinv([a[o + 3], a[o + 4], a[o + 5], a[o + 6]]), [b[o + 3], b[o + 4], b[o + 5], b[o + 6]]));
    v[j * 3] = axis[0] * angle;
    v[j * 3 + 1] = axis[1] * angle;
    v[j * 3 + 2] = axis[2] * angle;
  }
  for (let i = 0; i < 3; i++) v[J * 3 + i] = b[i] - a[i];
  return v;
}

/** Add a velocity offset dv scaled by s to a pose (in place): rotations in each joint's local frame. */
function addVelocity(local, dv, s) {
  const J = local.length / TF;
  for (let j = 0; j < J; j++) {
    const o = j * TF;
    const r = [dv[j * 3] * s, dv[j * 3 + 1] * s, dv[j * 3 + 2] * s];
    const a = Math.hypot(...r);
    if (a < 1e-9) continue;
    const q = qnormalize(qmul([local[o + 3], local[o + 4], local[o + 5], local[o + 6]], qFromAxisAngle(r.map((x) => x / a), a)));
    local[o + 3] = q[0];
    local[o + 4] = q[1];
    local[o + 5] = q[2];
    local[o + 6] = q[3];
  }
  for (let i = 0; i < 3; i++) local[i] += dv[J * 3 + i] * s;
}

export const DEFAULTS = Object.freeze({ limits: 1, feet: 1, floor: 1, balance: 0, smooth: 1 });

/**
 * Clamp a joint's rotation from rest to a cone of `limit` radians (in place,
 * with strength). Near 180 degrees the axis is ambiguous and would flip from
 * frame to frame; such a rotation is not anatomical anyway, so the joint
 * keeps its previous corrected rotation (`prev`, per joint) instead.
 */
function clampJoint(skeleton, local, j, limit, strength, prev = null) {
  const ref = tq(getTf(skeleton.reference, j));
  const cur = tq(getTf(local, j));
  const rel = qmul(qinv(ref), cur);
  const { axis, angle } = qToAxisAngle(rel);
  if (angle <= limit) {
    if (prev) prev[j] = cur;
    return false;
  }
  if (prev && angle > 150 * DEG && prev[j]) {
    setTf(local, j, tf(tp(getTf(local, j)), prev[j]));
    return true;
  }
  const target = angle - (angle - limit) * strength;
  const q = qnormalize(qmul(ref, qFromAxisAngle(axis, target)));
  setTf(local, j, tf(tp(getTf(local, j)), q));
  if (prev) prev[j] = q;
  return true;
}

/**
 * Joint limits on one pose, in place. Returns how many joints were corrected.
 * Order matters: limb shapes are recorded first, so clamping the spine or a
 * collarbone does not swing the arm attached to it; the limbs are then
 * re-aimed with true hinges, and the wrists and ankles clamped last.
 */
export function applyJointLimits(skeleton, local, strength = 1, cache = {}) {
  if (strength <= 0) return 0;
  cache.limbs ??= Object.values(LIMBS).map((n) => limbInfo(skeleton, n));
  cache.normals ??= cache.limbs.map(() => null);
  if (!cache.inner) {
    const ends = new Set(cache.limbs.map((l) => l.e));
    const all = Object.entries(JOINT_LIMITS).map(([n, d]) => [skeleton.jointIndex(n), d * DEG]).filter(([j]) => j >= 0);
    cache.inner = all.filter(([j]) => !ends.has(j));
    cache.ends = all.filter(([j]) => ends.has(j));
  }
  let n = 0;
  let world = skeleton.world(local);
  const P = (j) => [world[j * TF], world[j * TF + 1], world[j * TF + 2]];
  const dirs = cache.limbs.map((info) => {
    const [u, m, e] = [P(info.u), P(info.m), P(info.e)];
    return [[m[0] - u[0], m[1] - u[1], m[2] - u[2]], [e[0] - m[0], e[1] - m[1], e[2] - m[2]]];
  });
  cache.prev ??= {};
  cache.changed = [];
  for (const [j, limit] of cache.inner) if (clampJoint(skeleton, local, j, limit, strength, cache.prev)) (n++, cache.changed.push(j));
  world = skeleton.world(local);
  cache.limbs.forEach((info, i) => {
    const before = [info.u, info.m, info.e].map((j) => tq(getTf(local, j)));
    const inputAngle = hingeAngle(info, local);
    const r = solveLimb(skeleton, info, local, world, dirs[i][0], dirs[i][1], { prevNormal: cache.normals[i] });
    // Fail safe: a solve that would force a large clamp on a limb whose own
    // bend was already in range is a geometric corner case; keep the input
    // limb for this frame rather than swing the arm or leg.
    if (r.outBy > 0.35 && inputAngle > HINGE_RANGE[0] - 0.05 && inputAngle < HINGE_RANGE[1] + 0.05) {
      [info.u, info.m, info.e].forEach((j, k) => setTf(local, j, tf(tp(getTf(local, j)), before[k])));
      world = skeleton.world(local);
      return;
    }
    cache.normals[i] = r.normal;
    if (r.clamped) (n++, cache.changed.push(info.m));
    if (strength < 1) {
      // Partial strength: blend the limb's rotations back toward the original.
      [info.u, info.m, info.e].forEach((j, k) => {
        setTf(local, j, tf(tp(getTf(local, j)), slerp(before[k], tq(getTf(local, j)), strength)));
      });
    }
  });
  for (const [j, limit] of cache.ends) if (clampJoint(skeleton, local, j, limit, strength, cache.prev)) (n++, cache.changed.push(j));
  return n;
}

function slerp(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = d < 0 ? -1 : 1;
  d = Math.abs(d);
  if (d > 0.9995) return qnormalize(a.map((x, i) => x + (s * b[i] - x) * t));
  const th = Math.acos(d);
  const wa = Math.sin((1 - t) * th) / Math.sin(th);
  const wb = (s * Math.sin(t * th)) / Math.sin(th);
  return qnormalize(a.map((x, i) => wa * x + wb * b[i]));
}

/**
 * Stateful corrector for a stream of played frames (one dancer).
 */
export class Constraints {
  constructor(skeleton, settings = DEFAULTS) {
    this.sk = skeleton;
    this.settings = { ...DEFAULTS, ...settings };
    const idx = (n) => skeleton.jointIndex(n);
    // One hinge state per limb, shared by the limit pass and the foot IK (indices in LIMBS order).
    this.cache = {};
    applyJointLimits(skeleton, Float64Array.from(skeleton.reference), 1, this.cache);
    this.legs = [
      { info: this.cache.limbs[2], slot: 2, heel: 0, toe: 1, toeJ: idx('LeftToeBase'), lock: null, w: 0, side: null },
      { info: this.cache.limbs[3], slot: 3, heel: 2, toe: 3, toeJ: idx('RightToeBase'), lock: null, w: 0, side: null },
    ];
    this.cache.normals = this.cache.limbs.map(() => null);
    this.cache.prev = {};
    this.floorJoints = Object.entries(FLOOR_CLEARANCE).map(([n, c]) => [idx(n), c]);
    this.mass = Object.entries(MASS).map(([n, m]) => [idx(n), m]);
    this.lift = 0;
    this.stats = { limits: 0, locked: 0, lifted: 0, balanced: 0, frames: 0 };
  }

  set(settings) {
    this.settings = { ...this.settings, ...settings };
  }

  /**
   * Correct frames [from..to] of a segment in place.
   * poses: frames * joints * 7; contacts: frames * 4 (probabilities), may be null.
   * Returns, per frame from..to, what was done: { joints: corrected joints, planted: [left, right], lifted }.
   */
  apply(poses, contacts, from, to) {
    const J = this.sk.numJoints;
    // Handover: the velocity the body had minus the velocity the new segment
    // starts with, added as an offset that starts at zero and fades out
    // (dv * t * (1 - t/T)^2), so position and velocity are both continuous.
    // A new segment starts from the corrected body (its context was the
    // corrected frames), so the model's output already includes the
    // corrections made so far: re-base them to zero here, or they would be
    // applied twice (a pop at every handover). Planted feet keep their anchor.
    const activity = [];
    let dv = null;
    if (from > 0) {
      for (const leg of this.legs) {
        leg.offset = [0, 0, 0];
        leg.vel = [0, 0, 0];
      }
      this.lift = 0;
      // Handover velocity: what was shown, against how the new segment leaves
      // the last shown pose (its frame from-1 is that pose).
      if (this.shownLast && this.shownBeforeLast && this.settings.smooth > 0 && to >= from) {
        const before = velocity(this.shownBeforeLast, this.shownLast);
        // Compare corrected with corrected: the new frame gets the joint limits
        // (on a copy, with a copy of the hinge state) so removing a raw elbow
        // twist is not mistaken for motion.
        const next = Float64Array.from(poses.subarray(from * J * TF, (from + 1) * J * TF));
        if (this.settings.limits > 0) {
          const c = this.cache;
          applyJointLimits(this.sk, next, this.settings.limits, { limbs: c.limbs, inner: c.inner, ends: c.ends, normals: [...c.normals], prev: { ...c.prev } });
        }
        const after = velocity(poses.subarray((from - 1) * J * TF, from * J * TF), next);
        // At most 0.35 rad (and 3 cm) per frame of correction, per joint: enough
        // to ease a limb that reverses direction at the join.
        dv = before.map((x, i) => (x - after[i]) * this.settings.smooth);
        for (let j = 0; j < dv.length / 3; j++) {
          const m = Math.hypot(dv[j * 3], dv[j * 3 + 1], dv[j * 3 + 2]);
          const cap = j === dv.length / 3 - 1 ? 0.03 : 0.35;
          if (m > cap) for (let i = 0; i < 3; i++) dv[j * 3 + i] *= cap / m;
        }
      }
    }
    for (let f = from; f <= to; f++) {
      const local = poses.subarray(f * J * TF, (f + 1) * J * TF);
      const k = f - from + 1;
      if (dv && k <= HANDOVER_FRAMES) addVelocity(local, dv, k * (1 - k / (HANDOVER_FRAMES + 1)) ** 2);
      activity.push(this.frame(local, contacts ? contacts.subarray(f * 4, f * 4 + 4) : null));
      this.shownBeforeLast = this.shownLast;
      this.shownLast = Float64Array.from(local);
    }
    return activity;
  }

  frame(local, contact) {
    const S = this.settings;
    const sk = this.sk;
    this.stats.frames++;
    const act = { joints: [], planted: [false, false], lifted: false };
    if (S.limits > 0) {
      this.stats.limits += applyJointLimits(sk, local, S.limits, this.cache) > 0 ? 1 : 0;
      act.joints = this.cache.changed.slice();
    }

    // Floor: lift the body so nothing is below the floor (smoothed lift, so it does not jump).
    let world = sk.world(local);
    if (S.floor > 0) {
      let need = 0;
      for (const [j, c] of this.floorJoints) need = Math.max(need, c - world[j * TF + 1]);
      // Rise at once when needed, settle back slowly.
      this.lift = need > this.lift ? need : this.lift + (need - this.lift) * 0.15;
      if (this.lift > 1e-4) {
        local[1] += this.lift * S.floor;
        this.stats.lifted++;
        act.lifted = true;
        world = sk.world(local);
      }
    }

    // Balance: shift the hips so the centre of mass is over the support.
    const support = contact ? this.legs.filter((leg) => Math.max(contact[leg.heel], contact[leg.toe]) > 0.5) : [];
    if (S.balance > 0 && support.length) {
      let com = [0, 0];
      let total = 0;
      for (const [j, m] of this.mass) {
        com[0] += world[j * TF] * m;
        com[1] += world[j * TF + 2] * m;
        total += m;
      }
      com = com.map((x) => x / total);
      const feet = support.map((leg) => [world[leg.info.e * TF], world[leg.info.e * TF + 2]]);
      const c = [feet.reduce((s, p) => s + p[0], 0) / feet.length, feet.reduce((s, p) => s + p[1], 0) / feet.length];
      const radius = feet.length > 1 ? Math.hypot(feet[0][0] - feet[1][0], feet[0][1] - feet[1][1]) / 2 + 0.08 : 0.1;
      const d = Math.hypot(com[0] - c[0], com[1] - c[1]);
      if (d > radius) {
        const k = (Math.min(d - radius, 0.08) / d) * S.balance;
        local[0] += (c[0] - com[0]) * k;
        local[2] += (c[1] - com[1]) * k;
        this.stats.balanced++;
        world = sk.world(local);
      }
    }

    // Feet: plant the part in contact (heel: the ankle; toe only: the toe, the
    // foot pivoting on it). The contact signal is smoothed and needs to hold
    // for a few frames; a planted foot keeps its anchor through a heel-toe
    // roll; the correction follows a critically damped spring (no sudden
    // accelerations); a foot lets go when the body has moved too far from it.
    if (S.feet > 0) {
      for (const leg of this.legs) {
        const P = (j) => [world[j * TF], world[j * TF + 1], world[j * TF + 2]];
        const ankle = P(leg.info.e);
        const toePos = P(leg.toeJ);
        const speed = leg.lastAnkle ? Math.hypot(ankle[0] - leg.lastAnkle[0], ankle[2] - leg.lastAnkle[2]) * 30 : 0;
        leg.lastAnkle = ankle;
        // Where the knee points, tracked every frame (smoothed, kept on the
        // same side; a nearly straight knee's direction is noise and would
        // flip the bend plane (and twist the thigh).
        const hipNow = P(leg.info.u);
        const kneeNow = P(leg.info.m);
        const bend = [0, 1, 2].map((i) => kneeNow[i] - (hipNow[i] + ankle[i]) / 2);
        const bl = Math.hypot(...bend);
        if (!leg.side) leg.side = bl > 1e-6 ? bend.map((x) => x / bl) : [0, 0, 1];
        else if (bl > 0.04) {
          let cur = bend.map((x) => x / bl);
          if (cur[0] * leg.side[0] + cur[1] * leg.side[1] + cur[2] * leg.side[2] < 0 && bl < 0.08) cur = cur.map((x) => -x);
          const mix = leg.side.map((x, i) => x * 0.7 + cur[i] * 0.3);
          const ml = Math.hypot(...mix);
          if (ml > 1e-6) leg.side = mix.map((x) => x / ml);
        }
        let desired = [0, 0, 0];
        if (contact) {
          const raw = Math.max(contact[leg.heel], contact[leg.toe]);
          leg.p = leg.p === undefined ? raw : leg.p + (raw - leg.p) * 0.35; // smoothed contact
          leg.held = (leg.held || 0) + 1;
          if (leg.lock && leg.p < 0.3 && leg.held >= 6) {
            leg.lock = null;
            leg.held = 0;
          } else if (!leg.lock && leg.p > 0.7 && speed < 0.6 && leg.held >= 3) {
            const mode = contact[leg.heel] >= contact[leg.toe] ? 'heel' : 'toe';
            const at = mode === 'heel' ? ankle : toePos;
            leg.lock = { mode, at: at.map((x, k) => x + (leg.offset?.[k] || 0)) };
            leg.held = 0;
          }
          if (leg.lock) {
            const want = leg.lock.mode === 'heel' ? leg.lock.at : leg.lock.at.map((x, k) => x - (toePos[k] - ankle[k]));
            desired = want.map((x, k) => x - ankle[k]);
            if (Math.hypot(...desired) > 0.15) {
              leg.lock = null; // the body has moved on: let the foot go
              leg.held = 0;
              desired = [0, 0, 0];
            }
          }
        }
        desired = desired.map((x) => x * S.feet);
        // Critically damped spring toward the desired correction.
        leg.offset ??= [0, 0, 0];
        leg.vel ??= [0, 0, 0];
        const k = 0.12;
        const c = 2 * Math.sqrt(k);
        for (let i = 0; i < 3; i++) {
          leg.vel[i] += k * (desired[i] - leg.offset[i]) - c * leg.vel[i];
          leg.offset[i] += leg.vel[i];
        }
        if (Math.hypot(...leg.offset) < 1e-4) continue;
        const target = ankle.map((x, i) => x + leg.offset[i]);
        const hip = P(leg.info.u);
        const knee = P(leg.info.m);
        const { dU, dM } = ikDirections(leg.info, hip, target, leg.side);
        const r = solveLimb(sk, leg.info, local, world, dU, dM, { prevNormal: this.cache.normals[leg.slot] });
        this.cache.normals[leg.slot] = r.normal;
        this.stats.locked++;
      }
      this.legs.forEach((leg, i) => (act.planted[i] = !!leg.lock));
    }
    return act;
  }
}

export { HINGE_RANGE };
