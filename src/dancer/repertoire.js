// The repertoire: what the dancer has been taught.
//
// A clip is split into movements at natural pauses (low speed of hands, feet
// and head), each 0.6 to 2.5 s long. A movement keeps where it came from, a
// few key moments (the poses MotionBricks will pass through, about 0.9 s
// apart) and descriptors the improviser chooses by.

import { TF } from '../engine/math.js';

export const MIN_LEN = 0.6;
export const MAX_LEN = 2.5;
const KEY_SPACING = 0.9;
const FPS = 30;

const ENDS = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot', 'Head'];
export const PARTS = { handL: 'LeftHand', handR: 'RightHand', footL: 'LeftFoot', footR: 'RightFoot', head: 'Head' };

function heading(w, rl, ll) {
  const ax = w[rl * TF] - w[ll * TF];
  const az = w[rl * TF + 2] - w[ll * TF + 2];
  // forward = up x across
  return Math.atan2(az, -ax);
}

/**
 * Body shape for comparing poses: hands, feet and head relative to the hips'
 * ground point, in the body's own facing frame (x right... z forward), plus
 * hip height. 16 numbers.
 */
export function shapeOf(rel, h) {
  const c = Math.cos(h);
  const s = Math.sin(h);
  const out = [];
  for (const k of ENDS) {
    const [x, y, z] = rel[k];
    // Rotate by -h about Y so forward is +Z.
    out.push(c * x - s * z, y, s * x + c * z);
  }
  return out;
}

/** The same shape seen in a mirror (left and right swapped, x negated). */
export function mirrorShape(shape) {
  const part = (i) => shape.slice(i * 3, i * 3 + 3);
  const flip = ([x, y, z]) => [-x, y, z];
  // ENDS order: LeftHand, RightHand, LeftFoot, RightFoot, Head
  return [...flip(part(1)), ...flip(part(0)), ...flip(part(3)), ...flip(part(2)), ...flip(part(4))];
}

/** Per-frame features of a clip resampled at 30 Hz. */
export function analyse(clip, skeleton) {
  const J = skeleton.numJoints;
  const n = Math.max(2, Math.floor(clip.duration * FPS) + 1);
  const idx = Object.fromEntries([...ENDS, 'Hips', 'RightLeg', 'LeftLeg', 'LeftToeBase', 'RightToeBase'].map((k) => [k, skeleton.jointIndex(k)]));
  const local = new Float64Array(J * TF);
  const w = new Float64Array(J * TF);
  const rest = skeleton.world(skeleton.reference);
  const restHips = rest[idx.Hips * TF + 1];
  const frames = [];
  for (let f = 0; f < n; f++) {
    clip.sample(f / FPS, local);
    skeleton.world(local, w);
    const root = [w[idx.Hips * TF], w[idx.Hips * TF + 1], w[idx.Hips * TF + 2]];
    const rel = {};
    for (const k of ENDS) {
      const o = idx[k] * TF;
      rel[k] = [w[o] - root[0], w[o + 1], w[o + 2] - root[2]];
    }
    const h = heading(w, idx.RightLeg, idx.LeftLeg);
    frames.push({ root, rel, heading: h, shape: shapeOf(rel, h), toeY: Math.min(w[idx.LeftToeBase * TF + 1], w[idx.RightToeBase * TF + 1]) });
  }
  // Speed of the end effectors relative to the root (m/s), lightly smoothed.
  const raw = frames.map((fr, f) => {
    if (f === 0) return 0;
    const p = frames[f - 1];
    let s = 0;
    for (const k of ENDS) s += Math.hypot(fr.rel[k][0] - p.rel[k][0], fr.rel[k][1] - p.rel[k][1], fr.rel[k][2] - p.rel[k][2]);
    return (s * FPS) / ENDS.length;
  });
  raw[0] = raw[1] ?? 0;
  const speed = raw.map((_, i) => {
    let s = 0;
    let c = 0;
    for (let k = -3; k <= 3; k++) {
      const v = raw[i + k];
      if (v !== undefined) {
        s += v;
        c++;
      }
    }
    return s / c;
  });
  return { frames, speed, restHips };
}

/** Split a clip into movement time ranges [start, end] in seconds. */
export function segment(analysis) {
  const { speed } = analysis;
  const n = speed.length;
  const duration = (n - 1) / FPS;
  if (duration < MIN_LEN) return [];
  const minima = [];
  for (let i = 1; i < n - 1; i++) if (speed[i] <= speed[i - 1] && speed[i] <= speed[i + 1]) minima.push(i);
  const ranges = [];
  let t = 0;
  while (duration - t >= MIN_LEN) {
    if (duration - t <= MAX_LEN) {
      ranges.push([t, duration]);
      break;
    }
    const lo = Math.ceil((t + MIN_LEN) * FPS);
    const hi = Math.floor((t + MAX_LEN) * FPS);
    // The slowest local minimum in the window, else the slowest frame around 1.5 s.
    let best = -1;
    for (const m of minima) if (m >= lo && m <= hi && (best < 0 || speed[m] < speed[best])) best = m;
    if (best < 0) {
      const a = Math.ceil((t + 1.0) * FPS);
      const b = Math.floor((t + 2.0) * FPS);
      best = a;
      for (let i = a; i <= b; i++) if (speed[i] < speed[best]) best = i;
    }
    // Do not leave a tail shorter than MIN_LEN.
    if ((n - 1 - best) / FPS < MIN_LEN) {
      ranges.push([t, duration]);
      break;
    }
    ranges.push([t, best / FPS]);
    t = best / FPS;
  }
  return ranges;
}

/** Descriptors of the movement between two times. */
export function describe(analysis, start, end) {
  const { frames, speed, restHips } = analysis;
  const a = Math.round(start * FPS);
  const b = Math.min(frames.length - 1, Math.round(end * FPS));
  const span = frames.slice(a, b + 1);
  const sp = speed.slice(a, b + 1);
  const mean = (xs) => xs.reduce((s, x) => s + x, 0) / Math.max(1, xs.length);
  const dist = (v) => Math.hypot(v[0], v[1] - restHips, v[2]);
  // Path length of each part relative to the root: which part leads.
  const path = {};
  for (const [part, joint] of Object.entries(PARTS)) {
    let s = 0;
    for (let i = 1; i < span.length; i++) {
      const p = span[i - 1].rel[joint];
      const q = span[i].rel[joint];
      s += Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]);
    }
    path[part] = s;
  }
  const lead = Object.entries(path).sort((x, y) => y[1] - x[1])[0][0];
  let turn = 0;
  for (let i = 1; i < span.length; i++) {
    const d = span[i].heading - span[i - 1].heading;
    turn += Math.atan2(Math.sin(d), Math.cos(d));
  }
  const first = span[0].root;
  const last = span[span.length - 1].root;
  const headTop = (fr) => fr.rel.Head[1];
  return {
    duration: end - start,
    // 1 = standing hip height; lower is a lower level.
    level: mean(span.map((fr) => fr.root[1])) / restHips,
    lowest: Math.min(...span.map(headTop)),
    // Mean distance of hands and feet from the hips (m): size in the kinesphere.
    extension: mean(span.map((fr) => mean([fr.rel.LeftHand, fr.rel.RightHand, fr.rel.LeftFoot, fr.rel.RightFoot].map(dist)))),
    energy: mean(sp),
    peak: Math.max(...sp),
    stillness: mean(sp.map((v) => (v < 0.15 ? 1 : 0))),
    travel: Math.hypot(last[0] - first[0], last[2] - first[2]),
    turn,
    armsUp: mean(span.map((fr) => (fr.rel.LeftHand[1] > fr.rel.Head[1] || fr.rel.RightHand[1] > fr.rel.Head[1] ? 1 : 0))),
    // Both toes off the floor (rest toe height is about 0.02 m).
    airborne: mean(span.map((fr) => (fr.toeY > 0.1 ? 1 : 0))),
    startShape: span[0].shape,
    endShape: span[span.length - 1].shape,
    startHeading: span[0].heading,
    endHeading: span[span.length - 1].heading,
    lead,
    path,
  };
}

/**
 * Key moments of a movement for MotionBricks targets, about 0.9 s apart.
 * Interior keys move (within 0.25 s, keeping gaps of at least 0.6 s) to the
 * best nearby moment: feet on the floor, then slowest. Handing over in mid-air
 * or mid-swing is where transitions get sharp.
 */
export function keyTimes(start, end, analysis = null) {
  const n = Math.max(1, Math.round((end - start) / KEY_SPACING));
  const keys = Array.from({ length: n + 1 }, (_, i) => start + ((end - start) * i) / n);
  if (!analysis || n < 2) return keys;
  const { frames, speed } = analysis;
  const cost = (f) => (frames[f].toeY > 0.1 ? 10 : 0) + speed[f];
  for (let i = 1; i < n; i++) {
    const lo = Math.max(keys[i] - 0.25, keys[i - 1] + 0.6);
    const hi = Math.min(keys[i] + 0.25, keys[i + 1] - 0.6);
    if (hi < lo) continue;
    let best = Math.round(keys[i] * FPS);
    for (let f = Math.ceil(lo * FPS); f <= Math.floor(hi * FPS); f++) if (f < frames.length && cost(f) < cost(best)) best = f;
    keys[i] = best / FPS;
  }
  return keys;
}

/**
 * Teach the dancer a clip: returns its movements.
 * @param {{ id: string, name: string, clip: import('../engine/bvh.js').Clip, tags?: string[], credit?: string }} source
 */
export function learn(source, skeleton) {
  const analysis = analyse(source.clip, skeleton);
  return segment(analysis).map(([start, end], i) => ({
    id: `${source.id}#${i + 1}`,
    name: `${source.name} ${i + 1}`,
    source,
    start,
    end,
    keys: keyTimes(start, end, analysis),
    tags: source.tags || [],
    ...describe(analysis, start, end),
  }));
}
