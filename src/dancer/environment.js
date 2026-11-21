// The stage environment and whole-body interaction IK: interactive props
// (table, chair, glass, floor beacon), obstacle-aware navigation, seated
// lower-body layering, and precision arm/spine/head IK for reaching, grasping,
// drinking, and placing objects.
//
// Pure math (no three.js dependency here), so the director and tests run in
// Node while app/stage.js renders the 3D props in the browser.

import { TF, getTf, setTf, tf, tp, tq, qmul, qinv, qrot, qnormalize, qFromAxisAngle } from '../engine/math.js';
import { LIMBS, limbInfo, ikDirections, solveLimb } from '../engine/limbs.js';

const DEG = Math.PI / 180;
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function slerpQ(a, b, t) {
  if (t <= 0) return a;
  if (t >= 1) return b;
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = d < 0 ? -1 : 1;
  d = Math.abs(d);
  if (d > 0.9995) return qnormalize(a.map((x, i) => x + (s * b[i] - x) * t));
  const th = Math.acos(d);
  const wa = Math.sin((1 - t) * th) / Math.sin(th);
  const wb = (s * Math.sin(t * th)) / Math.sin(th);
  return qnormalize(a.map((x, i) => wa * x + wb * b[i]));
}

export class StageEnvironment {
  constructor() {
    this.reset();
  }

  reset() {
    // Table is at stage-left-upstage; chair is placed in front of it facing +Z (toward the table).
    this.table = {
      id: 'table',
      label: 'Table',
      x: 1.35,
      z: -0.78,
      y: 0,
      height: 0.72,
      radius: 0.42,
      standAnchor: { x: 1.35, z: -1.08, heading: 0 },
    };
    this.chair = {
      id: 'chair',
      label: 'Chair',
      x: 1.35,
      z: -1.45,
      y: 0,
      seatY: 0.53,
      heading: 0, // facing +Z toward the table
      radius: 0.28,
      approachAnchor: { x: 1.35, z: -1.08, heading: 0 },
      seatAnchor: { x: 1.35, y: 0.53, z: -1.45, heading: 0 },
      feet: {
        left: [1.45, 0.08, -1.08],
        right: [1.25, 0.08, -1.08],
      },
    };
    this.glass = {
      id: 'glass',
      label: 'Glass',
      x: 1.24,
      y: 0.76,
      z: -0.98,
      home: [1.24, 0.76, -0.98],
      heldBy: null, // 'RightHand' | null
      tilt: 0, // radians
      sips: 0,
    };
    this.beacon = {
      active: false,
      x: 0,
      z: 0,
      label: '',
    };
    this.seated = false;
    this.holding = null; // 'glass' | null
  }

  /** Resolve a named target or {x, z, heading} object into an approach destination { x, z, heading, label, targetId }. */
  resolveTarget(target, currentPlace = { x: 0, z: 0, heading: 0 }) {
    if (target && typeof target === 'object' && Number.isFinite(target.x) && Number.isFinite(target.z)) {
      const dx = target.x - currentPlace.x;
      const dz = target.z - currentPlace.z;
      const h = Number.isFinite(target.heading) ? target.heading : Math.hypot(dx, dz) > 0.05 ? Math.atan2(dx, dz) : currentPlace.heading;
      return { x: target.x, z: target.z, heading: h, label: target.label || `(${target.x.toFixed(1)}, ${target.z.toFixed(1)})`, targetId: target.targetId || 'point' };
    }
    const name = String(target || 'table').toLowerCase().trim();
    if (/\b(table|desk|counter|bar|glass|mug|cup|drink|water|soda|coffee)\b/.test(name)) {
      const a = this.table.standAnchor;
      return { x: a.x, z: a.z, heading: a.heading, label: 'the table', targetId: 'table' };
    }
    if (/\b(chair|seat|stool|bench)\b/.test(name)) {
      const a = this.chair.approachAnchor;
      return { x: a.x, z: a.z, heading: a.heading, label: 'the chair', targetId: 'chair' };
    }
    if (/\b(left|stage left)\b/.test(name)) {
      return { x: 1.6, z: 0.4, heading: Math.PI / 2, label: 'stage left', targetId: 'left' };
    }
    if (/\b(right|stage right)\b/.test(name)) {
      return { x: -1.6, z: 0.4, heading: -Math.PI / 2, label: 'stage right', targetId: 'right' };
    }
    if (/\b(front|downstage|camera|audience|forward)\b/.test(name)) {
      return { x: 0, z: 1.6, heading: 0, label: 'downstage', targetId: 'front' };
    }
    if (/\b(back|upstage)\b/.test(name)) {
      return { x: 0, z: -1.6, heading: Math.PI, label: 'upstage', targetId: 'back' };
    }
    return { x: 0, z: 0, heading: 0, label: 'centre stage', targetId: 'center' };
  }

  /**
   * Plan obstacle-free turtle moves ([{ move: 'turn'|'forward', ... }]) from `start`
   * ({ x, z, heading }) to `dest` ({ x, z, heading }). Avoids walking through the table.
   */
  planMovesTo(start, dest) {
    const points = [];
    // Check if the straight segment from start to dest intersects the table cylinder.
    const obs = { x: this.table.x, z: this.table.z, r: this.table.radius + 0.22 };
    const dx = dest.x - start.x;
    const dz = dest.z - start.z;
    const segLen = Math.hypot(dx, dz);
    if (segLen > 0.15) {
      const t = Math.max(0, Math.min(1, ((obs.x - start.x) * dx + (obs.z - start.z) * dz) / (segLen * segLen)));
      const cx = start.x + t * dx;
      const cz = start.z + t * dz;
      const distToObs = Math.hypot(cx - obs.x, cz - obs.z);
      const endDist = Math.hypot(dest.x - obs.x, dest.z - obs.z);
      const startDist = Math.hypot(start.x - obs.x, start.z - obs.z);
      if (t > 0.05 && t < 0.95 && distToObs < obs.r && endDist > obs.r * 0.8 && startDist > obs.r * 0.8) {
        // Detour tangentially around the table on the clearer side.
        const nx = -dz / segLen;
        const nz = dx / segLen;
        const side = (cx - obs.x) * nx + (cz - obs.z) * nz >= 0 ? 1 : -1;
        points.push({ x: obs.x + side * nx * (obs.r + 0.15), z: obs.z + side * nz * (obs.r + 0.15) });
      }
    }
    points.push({ x: dest.x, z: dest.z });

    const moves = [];
    let curX = start.x;
    let curZ = start.z;
    let curH = start.heading;
    for (const p of points) {
      const sx = p.x - curX;
      const sz = p.z - curZ;
      const d = Math.hypot(sx, sz);
      if (d > 0.08) {
        const wantH = Math.atan2(sx, sz);
        const turnDeg = (wrapAngle(wantH - curH) * 180) / Math.PI;
        if (Math.abs(turnDeg) >= 2) moves.push({ move: 'turn', angle: +turnDeg.toFixed(1), distance: null, radius: null });
        moves.push({ move: 'forward', distance: +d.toFixed(2), angle: null, radius: null });
        curH = wantH;
        curX = p.x;
        curZ = p.z;
      }
    }
    if (Number.isFinite(dest.heading)) {
      const finalTurn = (wrapAngle(dest.heading - curH) * 180) / Math.PI;
      if (Math.abs(finalTurn) >= 3) moves.push({ move: 'turn', angle: +finalTurn.toFixed(1), distance: null, radius: null });
    }
    return moves;
  }

  /** Update prop transforms from the actor's current local pose (e.g. glass held in RightHand). */
  updatePropsFromPose(skeleton, local) {
    if (this.glass.heldBy === 'RightHand') {
      const w = skeleton.world(local);
      const j = skeleton.jointIndex('RightHand');
      const m = skeleton.jointIndex('RightHandMiddle1');
      const handPos = [w[j * TF], w[j * TF + 1], w[j * TF + 2]];
      const midPos = m >= 0 ? [w[m * TF], w[m * TF + 1], w[m * TF + 2]] : handPos;
      // Offset slightly toward the palm center.
      this.glass.x = handPos[0] * 0.4 + midPos[0] * 0.6;
      this.glass.y = handPos[1] * 0.4 + midPos[1] * 0.6;
      this.glass.z = handPos[2] * 0.4 + midPos[2] * 0.6;
    }
  }
}

/**
 * Pose a frame into a seated posture on `chair` with blend weight `w` (0 = standing, 1 = seated).
 * Solves two-bone leg IK so both feet stay planted flat on the floor while the pelvis rests on
 * the chair seat and knees bend naturally at ~100 degrees.
 */
export function applySeatedLayer(skeleton, local, chair, w = 1, { leftArmRest = true, cache = {} } = {}) {
  if (w <= 0) return;
  cache.legL ??= limbInfo(skeleton, LIMBS.legL);
  cache.legR ??= limbInfo(skeleton, LIMBS.legR);
  cache.armL ??= limbInfo(skeleton, LIMBS.armL);
  const s = w * w * (3 - 2 * w); // smoothstep

  // Pelvis position and upright facing toward the table (chair.heading).
  const seat = chair.seatAnchor;
  local[0] = local[0] + (seat.x - local[0]) * s;
  local[1] = local[1] + (seat.y - local[1]) * s;
  local[2] = local[2] + (seat.z - local[2]) * s;

  // Align root yaw toward chair.heading and keep pelvis level.
  const refRootQ = tq(getTf(skeleton.reference, 0));
  const seatRootQ = qnormalize(qmul(qFromAxisAngle([0, 1, 0], seat.heading), refRootQ));
  const curRootQ = tq(getTf(local, 0));
  setTf(local, 0, tf([local[0], local[1], local[2]], slerpQ(curRootQ, seatRootQ, s)));

  // Calm the spine toward upright rest before any upper-body action layer runs.
  for (const name of ['Spine1', 'Spine2', 'Chest']) {
    const j = skeleton.jointIndex(name);
    if (j < 0) continue;
    const ref = getTf(skeleton.reference, j);
    const cur = getTf(local, j);
    setTf(local, j, tf(tp(cur), slerpQ(tq(cur), tq(ref), s * 0.75)));
  }

  const world = skeleton.world(local);
  const P = (j) => [world[j * TF], world[j * TF + 1], world[j * TF + 2]];
  const Q = (j) => [world[j * TF + 3], world[j * TF + 4], world[j * TF + 5], world[j * TF + 6]];

  // Solve both legs so ankles stay planted on the floor in front of the chair.
  const fwd = [Math.sin(seat.heading), 0.6, Math.cos(seat.heading)];
  const refWorld = (cache.refWorld ??= skeleton.world(skeleton.reference));
  for (const [info, footTarget] of [[cache.legL, chair.feet.left], [cache.legR, chair.feet.right]]) {
    const curAnkle = P(info.e);
    const targetAnkle = [
      curAnkle[0] + (footTarget[0] - curAnkle[0]) * s,
      curAnkle[1] + (footTarget[1] - curAnkle[1]) * s,
      curAnkle[2] + (footTarget[2] - curAnkle[2]) * s,
    ];
    const hip = P(info.u);
    const { dU, dM } = ikDirections(info, hip, targetAnkle, fwd);
    const refFootQ = qnormalize(qmul(qFromAxisAngle([0, 1, 0], seat.heading), tq(getTf(refWorld, info.e))));
    setTf(world, info.e, tf(P(info.e), slerpQ(Q(info.e), refFootQ, s)));
    solveLimb(skeleton, info, local, world, dU, dM);
  }

  // Rest left hand naturally near the left thigh / lap while seated.
  if (leftArmRest) {
    const curHandL = P(cache.armL.e);
    const lapL = [seat.x + 0.16, seat.y + 0.08, seat.z + 0.25];
    const targetL = [
      curHandL[0] + (lapL[0] - curHandL[0]) * s * 0.85,
      curHandL[1] + (lapL[1] - curHandL[1]) * s * 0.85,
      curHandL[2] + (lapL[2] - curHandL[2]) * s * 0.85,
    ];
    const { dU, dM } = ikDirections(cache.armL, P(cache.armL.u), targetL, [0.5, -1, -0.3]);
    solveLimb(skeleton, cache.armL, local, world, dU, dM);
  }
}

/**
 * Apply upper-body spine lean, head pitch, and two-bone right-arm (or left-arm) IK
 * toward `targetPos` in world space with weight `w` (0..1).
 */
export function applyArmReachIK(skeleton, local, targetPos, w = 1, {
  hand = 'right',
  spineLean = 0, // radians forward lean
  headPitch = 0, // radians head tilt (+ = tilt back to drink, - = look down at glass)
  cache = {},
} = {}) {
  if (w <= 0) return;
  const s = w * w * (3 - 2 * w);
  if (Math.abs(spineLean) > 1e-4) {
    for (const [name, share] of [['Spine1', 0.45], ['Spine2', 0.35], ['Chest', 0.20]]) {
      const j = skeleton.jointIndex(name);
      if (j < 0) continue;
      const cur = getTf(local, j);
      // In SOMA's spine frame (+X up along spine, +Y forward, +Z left), rotation about Z bends forward/back.
      const leanQ = qFromAxisAngle([0, 0, 1], -spineLean * share * s);
      setTf(local, j, tf(tp(cur), qnormalize(qmul(tq(cur), leanQ))));
    }
  }
  if (Math.abs(headPitch) > 1e-4) {
    for (const [name, share] of [['Neck1', 0.45], ['Head', 0.55]]) {
      const j = skeleton.jointIndex(name);
      if (j < 0) continue;
      const cur = getTf(local, j);
      const tiltQ = qFromAxisAngle([0, 0, 1], headPitch * share * s);
      setTf(local, j, tf(tp(cur), qnormalize(qmul(tq(cur), tiltQ))));
    }
  }

  const world = skeleton.world(local);
  const P = (j) => [world[j * TF], world[j * TF + 1], world[j * TF + 2]];
  const arm = hand === 'left'
    ? (cache.armL ??= limbInfo(skeleton, LIMBS.armL))
    : (cache.armR ??= limbInfo(skeleton, LIMBS.armR));
  const curHand = P(arm.e);
  const goal = [
    curHand[0] + (targetPos[0] - curHand[0]) * s,
    curHand[1] + (targetPos[1] - curHand[1]) * s,
    curHand[2] + (targetPos[2] - curHand[2]) * s,
  ];
  const shoulder = P(arm.u);
  const pole = hand === 'left' ? [0.6, -1, -0.25] : [-0.6, -1, -0.25];
  const { dU, dM } = ikDirections(arm, shoulder, goal, pole);
  solveLimb(skeleton, arm, local, world, dU, dM);
}

/** Mouth position in world space for drinking, from the current pose's Head and HeadEnd joints. */
export function mouthPosition(skeleton, local) {
  const w = skeleton.world(local);
  const h = skeleton.jointIndex('Head') * TF;
  const e = skeleton.jointIndex('HeadEnd') * TF;
  const rl = skeleton.jointIndex('RightLeg') * TF;
  const ll = skeleton.jointIndex('LeftLeg') * TF;
  const heading = Math.atan2(w[rl + 2] - w[ll + 2], -(w[rl] - w[ll]));
  const fx = Math.sin(heading);
  const fz = Math.cos(heading);
  return [
    (w[h] * 0.65 + w[e] * 0.35) + fx * 0.14,
    (w[h + 1] * 0.65 + w[e + 1] * 0.35) - 0.02,
    (w[h + 2] * 0.65 + w[e + 2] * 0.35) + fz * 0.14,
  ];
}
