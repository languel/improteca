// Retarget a BVH clip from another skeleton (e.g. CMU mocap) onto SOMA.
//
// Joint names cannot be trusted across skeletons (CMU's "LeftLeg" is the
// shin; SOMA's is the thigh), so joints are matched through an explicit map.
// Method: calibrated world-rotation transfer.
//   1. Calibrate: pose SOMA's rest skeleton so every mapped bone points the
//      same way as the source's rest bone (swing only), giving A_j.
//   2. Each frame: world_soma(j) = world_src(c) * world_src_rest(c)^-1 * A_j.
//      This carries the source's swing and twist; unmapped SOMA joints (the
//      fingers) keep their rest rotation relative to their parent.
//   3. Root position: the source hips, scaled by the ratio of leg lengths,
//      then shifted so the feet meet the floor as in SOMA's rest pose.
// The result is resampled to 30 Hz.

import { TF, getTf, setTf, tf, tp, tq, tmul, qmul, qinv, qnormalize, qrot, cross } from './math.js';
import { Clip } from './bvh.js';

/** SOMA joint <- CMU joint (as in the CMU BVH exports, e.g. una-dinosauria/cmu-mocap). */
export const CMU_TO_SOMA = {
  Hips: 'Hips',
  Spine1: 'LowerBack',
  Spine2: 'Spine',
  Chest: 'Spine1',
  Neck1: 'Neck',
  Neck2: 'Neck1',
  Head: 'Head',
  LeftShoulder: 'LeftShoulder',
  LeftArm: 'LeftArm',
  LeftForeArm: 'LeftForeArm',
  LeftHand: 'LeftHand',
  RightShoulder: 'RightShoulder',
  RightArm: 'RightArm',
  RightForeArm: 'RightForeArm',
  RightHand: 'RightHand',
  LeftLeg: 'LeftUpLeg',
  LeftShin: 'LeftLeg',
  LeftFoot: 'LeftFoot',
  LeftToeBase: 'LeftToeBase',
  RightLeg: 'RightUpLeg',
  RightShin: 'RightLeg',
  RightFoot: 'RightFoot',
  RightToeBase: 'RightToeBase',
};

/** For calibration: each mapped SOMA joint's bone points toward this child. */
const BONE_CHILD = {
  Spine1: 'Spine2', Spine2: 'Chest', Chest: 'Neck1', Neck1: 'Neck2', Neck2: 'Head',
  LeftShoulder: 'LeftArm', LeftArm: 'LeftForeArm', LeftForeArm: 'LeftHand', LeftHand: 'LeftHandMiddle1',
  RightShoulder: 'RightArm', RightArm: 'RightForeArm', RightForeArm: 'RightHand', RightHand: 'RightHandMiddle1',
  LeftLeg: 'LeftShin', LeftShin: 'LeftFoot', LeftFoot: 'LeftToeBase', LeftToeBase: 'LeftToeEnd',
  RightLeg: 'RightShin', RightShin: 'RightFoot', RightFoot: 'RightToeBase', RightToeBase: 'RightToeEnd',
};

/** The shortest rotation taking direction a to direction b. */
function rotationBetween(a, b) {
  const na = Math.hypot(...a);
  const nb = Math.hypot(...b);
  if (na < 1e-9 || nb < 1e-9) return [0, 0, 0, 1];
  const u = a.map((x) => x / na);
  const v = b.map((x) => x / nb);
  const d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  if (d < -0.999999) {
    let axis = cross([1, 0, 0], u);
    if (Math.hypot(...axis) < 1e-6) axis = cross([0, 1, 0], u);
    const n = Math.hypot(...axis);
    return [axis[0] / n, axis[1] / n, axis[2] / n, 0];
  }
  const c = cross(u, v);
  return qnormalize([c[0], c[1], c[2], 1 + d]);
}

/** Rotation taking an orthonormal-ish frame (primary a1, secondary a2) onto (b1, b2). */
function rotationBetweenFrames(a1, a2, b1, b2) {
  const q1 = rotationBetween(a1, b1);
  const a2r = qrot(q1, a2);
  // Twist about b1 so a2r lines up with b2 in the plane perpendicular to b1.
  const n = b1.map((x) => x / Math.hypot(...b1));
  const proj = (v) => {
    const d = v[0] * n[0] + v[1] * n[1] + v[2] * n[2];
    return [v[0] - d * n[0], v[1] - d * n[1], v[2] - d * n[2]];
  };
  const q2 = rotationBetween(proj(a2r), proj(b2));
  return qmul(q2, q1);
}

/**
 * @param {Clip} source  a clip on its own file skeleton (parseBVH without a target skeleton)
 * @param {import('./skeleton.js').Skeleton} soma
 * @param {Record<string,string>} map  SOMA joint -> source joint
 * @param {{ sampleRate?: number }} [options]
 */
export function retarget(source, soma, map = CMU_TO_SOMA, { sampleRate = 30 } = {}) {
  const src = source.skeleton;
  const J = soma.numJoints;
  const S = src.numJoints;
  const pairs = [];
  for (const [s, c] of Object.entries(map)) {
    const sj = soma.jointIndex(s);
    const cj = src.jointIndex(c);
    if (sj < 0) throw new Error(`SOMA has no joint ${s}`);
    if (cj < 0) throw new Error(`The clip has no joint ${c} (for ${s})`);
    pairs.push([sj, cj]);
  }
  const srcOf = new Int32Array(J).fill(-1);
  for (const [sj, cj] of pairs) srcOf[sj] = cj;

  // ---- calibration ----------------------------------------------------------
  const srcRest = src.world(src.reference); // identity rotations: offsets only
  const pos = (w, j) => [w[j * TF], w[j * TF + 1], w[j * TF + 2]];
  const rot = (w, j) => [w[j * TF + 3], w[j * TF + 4], w[j * TF + 5], w[j * TF + 6]];
  const srcDir = (sj, childName) => {
    const c = srcOf[soma.jointIndex(childName)];
    if (c < 0) return null;
    const a = pos(srcRest, srcOf[sj]);
    const b = pos(srcRest, c);
    return [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  };

  // Walk SOMA's hierarchy, rotating each mapped bone to the source's rest direction.
  const aligned = Float64Array.from(soma.reference); // local transforms
  let world = soma.world(aligned);
  const hips = soma.jointIndex('Hips');
  {
    // Hips: match the up (hips -> chest) and across (right leg -> left leg) frame.
    const up = (w, a, b) => { const p = pos(w, a); const q = pos(w, b); return [q[0] - p[0], q[1] - p[1], q[2] - p[2]]; };
    const sUp = up(world, hips, soma.jointIndex('Chest'));
    const sAcross = up(world, soma.jointIndex('RightLeg'), soma.jointIndex('LeftLeg'));
    const cUp = up(srcRest, srcOf[hips], srcOf[soma.jointIndex('Chest')]);
    const cAcross = up(srcRest, srcOf[soma.jointIndex('RightLeg')], srcOf[soma.jointIndex('LeftLeg')]);
    const q = rotationBetweenFrames(sUp, sAcross, cUp, cAcross);
    const h = getTf(aligned, hips);
    setTf(aligned, hips, tf(tp(h), qnormalize(qmul(q, tq(h)))));
    world = soma.world(aligned);
  }
  for (let j = 1; j < J; j++) {
    const name = soma.names[j];
    const child = BONE_CHILD[name];
    if (srcOf[j] < 0 || !child) continue;
    const target = srcDir(j, child);
    const k = soma.jointIndex(child);
    if (!target || k < 0) continue;
    const a = pos(world, j);
    const b = pos(world, k);
    const q = rotationBetween([b[0] - a[0], b[1] - a[1], b[2] - a[2]], target);
    // Apply q in world space at joint j: new local = parentWorld^-1 * q * world_j.
    const parentQ = rot(world, soma.parents[j]);
    const newWorldQ = qmul(q, rot(world, j));
    const l = getTf(aligned, j);
    setTf(aligned, j, tf(tp(l), qnormalize(qmul(qinv(parentQ), newWorldQ))));
    world = soma.world(aligned);
  }
  const alignedWorld = world;

  // ---- scale and floor --------------------------------------------------------
  const legLength = (w, top, bottom) => {
    const a = pos(w, top);
    const b = pos(w, bottom);
    return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  };
  const L = (n) => soma.jointIndex(n);
  const somaLeg = legLength(alignedWorld, L('LeftLeg'), L('LeftShin')) + legLength(alignedWorld, L('LeftShin'), L('LeftFoot'));
  const srcLeg = legLength(srcRest, srcOf[L('LeftLeg')], srcOf[L('LeftShin')]) + legLength(srcRest, srcOf[L('LeftShin')], srcOf[L('LeftFoot')]);
  const scale = somaLeg / srcLeg;

  // ---- frames -----------------------------------------------------------------
  const duration = source.duration;
  const frames = Math.max(1, Math.floor(duration * sampleRate) + 1);
  const poses = new Float64Array(frames * J * TF);
  const srcLocal = new Float64Array(S * TF);
  const srcWorld = new Float64Array(S * TF);
  const restInv = pairs.map(([, cj]) => qinv(rot(srcRest, cj))); // identity for BVH rest, kept for generality
  const outWorld = new Float64Array(J * TF);
  for (let f = 0; f < frames; f++) {
    source.sample(f / sampleRate, srcLocal);
    src.world(srcLocal, srcWorld);
    // World rotations of mapped joints.
    const worldQ = new Array(J);
    pairs.forEach(([sj, cj], i) => {
      worldQ[sj] = qnormalize(qmul(qmul(rot(srcWorld, cj), restInv[i]), rot(alignedWorld, sj)));
    });
    // Locals in hierarchy order.
    for (let j = 0; j < J; j++) {
      const p = soma.parents[j];
      if (j === hips) {
        const hp = pos(srcWorld, srcOf[hips]);
        setTf(outWorld, j, tf([hp[0] * scale, hp[1] * scale, hp[2] * scale], worldQ[j]));
        setTf(poses, f * J + j, getTf(outWorld, j));
        continue;
      }
      const ref = getTf(aligned, j);
      const parentW = getTf(outWorld, p);
      let local;
      if (worldQ[j]) local = tf(tp(ref), qnormalize(qmul(qinv(tq(parentW)), worldQ[j])));
      else local = tf(tp(ref), tq(getTf(soma.reference, j)));
      setTf(poses, f * J + j, local);
      setTf(outWorld, j, tmul(parentW, local));
    }
  }

  // Floor: the lowest toe over the clip sits at SOMA's rest toe height.
  const clip = new Clip(soma, sampleRate, poses);
  const toes = [L('LeftToeBase'), L('RightToeBase'), L('LeftFoot'), L('RightFoot')];
  const restMin = Math.min(...toes.map((j) => soma.world(soma.reference)[j * TF + 1]));
  let low = Infinity;
  const w = new Float64Array(J * TF);
  for (let f = 0; f < frames; f++) {
    soma.world(poses.subarray(f * J * TF, (f + 1) * J * TF), w);
    for (const j of toes) low = Math.min(low, w[j * TF + 1]);
  }
  const lift = restMin - low;
  for (let f = 0; f < frames; f++) poses[(f * J + hips) * TF + 1] += lift;
  return clip;
}

/** Is this skeleton CMU-like (so CMU_TO_SOMA applies)? */
export function looksLikeCMU(skeleton) {
  return ['LowerBack', 'LeftUpLeg', 'LHipJoint'].every((n) => skeleton.jointIndex(n) >= 0);
}

/** Is this skeleton already SOMA (copy joints by name)? */
export function looksLikeSOMA(skeleton) {
  return ['Hips', 'Spine1', 'Chest', 'LeftShin', 'LeftToeBase', 'Neck2'].every((n) => skeleton.jointIndex(n) >= 0);
}

/**
 * Load any BVH as a SOMA clip: SOMA files are copied by joint name, CMU-style
 * files are retargeted. Returns { clip, method }.
 */
export function bvhToSoma(text, soma, parseBVH) {
  const raw = parseBVH(text);
  if (looksLikeSOMA(raw.skeleton)) return { clip: raw.remap(soma), method: 'SOMA' };
  if (looksLikeCMU(raw.skeleton)) return { clip: retarget(raw, soma), method: 'CMU retarget' };
  throw new Error('Unknown skeleton: expected SOMA or CMU joint names');
}
