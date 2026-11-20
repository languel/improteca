// The improviser: decides what to dance next, one movement at a time.
//
// For each step it scores every way of dancing every movement it knows
// (as taught or mirrored, forwards or backwards) against:
//   - the current intentions (from a prompt, a score or a partner),
//   - continuity: how close the movement's first pose is to where the body is,
//   - novelty: how recently that movement was danced,
// then picks one with seeded chance (variety sets the temperature), places
// it relative to the dancer (turning back toward the centre near the edge of
// the stage), sets its tempo, and turns its key moments into MotionBricks
// targets. Every choice is logged with its options, scores and reasons.

import { TF, tf, tmul, qFromAxisAngle } from '../engine/math.js';
import { SAMPLE_RATE, POSES_PER_TOKEN, MIN_TOKENS, MAX_TOKENS } from '../engine/soma.js';
import { makeMirror, placePoses } from './transforms.js';
import { mirrorShape, shapeOf, PARTS } from './repertoire.js';
import { Composer } from './composer.js';

/** Seeded random numbers (mulberry32). */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Defaults: no preference on any quality; moderate variety; natural tempo. */
export const NEUTRAL = Object.freeze({
  energy: null, // 0 calm .. 1 vigorous
  level: null, // 0 low .. 1 high
  size: null, // 0 small, close .. 1 large, extended
  travel: null, // 0 on the spot .. 1 travelling
  turn: null, // 0 no turning .. 1 turning
  armsUp: null, // 0 .. 1 arms overhead
  jump: null, // 0 .. 1 airborne
  stillness: null, // 0 .. 1 pauses and holds
  focus: null, // 'handL' | 'handR' | 'footL' | 'footR' | 'head' | 'arms' | 'legs'
  tags: [], // preferred material, e.g. ['ballet'] or ['contemporary']
  avoid: [], // material to avoid
  tempo: 1, // > 1 faster, < 1 slower
  variety: 0.5, // 0 repetitive .. 1 unpredictable
  mirror: 0.35, // chance weight for mirrored versions
  retrograde: 0.1, // chance weight for movements danced backwards
});

const QUALITIES = ['energy', 'level', 'size', 'travel', 'turn', 'armsUp', 'jump', 'stillness'];
const FEATURE = { energy: 'energy', level: 'level', size: 'extension', travel: 'travel', turn: 'absTurn', armsUp: 'armsUp', jump: 'airborne', stillness: 'stillness' };
const WORDS = {
  energy: ['calm', 'vigorous'], level: ['low', 'high'], size: ['small', 'expansive'], travel: ['on the spot', 'travelling'],
  turn: ['facing', 'turning'], armsUp: ['arms low', 'arms high'], jump: ['grounded', 'airborne'], stillness: ['continuous', 'still'],
};
const STAGE_RADIUS = 3;

/** Rank-normalise each feature across the repertoire to 0..1. */
function normalise(movements) {
  const keys = Object.values(FEATURE);
  for (const m of movements) m.absTurn = Math.abs(m.turn);
  const rank = {};
  for (const k of keys) {
    const sorted = movements.map((m) => m[k]).sort((a, b) => a - b);
    rank[k] = (v) => {
      let lo = 0;
      let hi = sorted.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (sorted[mid] < v) lo = mid + 1;
        else hi = mid;
      }
      return sorted.length > 1 ? lo / (sorted.length - 1) : 0.5;
    };
  }
  for (const m of movements) m.norm = Object.fromEntries(keys.map((k) => [k, rank[k](m[k])]));
}

function shapeDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s / (a.length / 3));
}

const mirrorPart = (p) => ({ handL: 'handR', handR: 'handL', footL: 'footR', footR: 'footL' })[p] || p;

export class Improviser {
  /**
   * @param {object} opts
   * @param {import('../engine/skeleton.js').Skeleton} opts.skeleton
   * @param {object[]} opts.movements  from learn()
   * @param {number} [opts.seed]
   */
  constructor({ skeleton, movements, seed = 1 }) {
    this.skeleton = skeleton;
    this.J = skeleton.numJoints;
    this.movements = movements;
    normalise(movements);
    this.seed = seed;
    this.random = rng(seed);
    this.intentions = { ...NEUTRAL };
    this.recent = [];
    this.log = [];
    this.mirror = makeMirror(skeleton);
    this.step = 0;
    this.composer = new Composer();
  }

  /** Phrase memory: motifs and the form so far (for display). */
  memory() {
    const name = (id) => this.movements.find((m) => m.id === id)?.name || id;
    return {
      motifs: this.composer.motifs.map((mo) => ({ name: mo.name, items: mo.items.map((it) => name(it.id)) })),
      form: this.composer.form.slice(),
    };
  }

  /** Learn more movements (e.g. while clips are still loading). */
  addMovements(movements) {
    this.movements = this.movements.concat(movements);
    normalise(this.movements);
  }

  setIntentions(intentions) {
    this.intentions = { ...NEUTRAL, ...intentions };
  }

  /** Score one candidate; returns { score, reasons }. */
  score(m, variant, here) {
    const I = this.intentions;
    let score = 0;
    const reasons = [];
    for (const q of QUALITIES) {
      const want = I[q];
      if (want === null || want === undefined) continue;
      const have = m.norm[FEATURE[q]];
      const fit = 1 - Math.abs(have - want); // 0..1
      score += 3 * (fit - 0.5);
      if (fit > 0.75) reasons.push(WORDS[q][want >= 0.5 ? 1 : 0]);
    }
    if (I.focus) {
      const lead = variant.mirrored ? mirrorPart(m.lead) : m.lead;
      const ok = I.focus === lead || (I.focus === 'arms' && lead.startsWith('hand')) || (I.focus === 'legs' && lead.startsWith('foot'));
      if (ok) {
        score += 1.5;
        reasons.push(`led by the ${({ handL: 'left hand', handR: 'right hand', footL: 'left foot', footR: 'right foot', head: 'head' })[lead]}`);
      }
    }
    const tags = m.tags || [];
    if (I.tags?.length) {
      const hit = I.tags.some((t) => tags.includes(t));
      score += hit ? 1.5 : -1.5;
      if (hit) reasons.push(I.tags.filter((t) => tags.includes(t)).join(', '));
    }
    // Continuity: a close first pose makes a natural transition.
    if (here) {
      let start = variant.retro ? m.endShape : m.startShape;
      if (variant.mirrored) start = mirrorShape(start);
      const d = shapeDistance(here, start); // metres, typically 0.05 .. 0.6
      score += 2 * (0.25 - d) * 4;
      if (d < 0.15) reasons.push('flows from the current pose');
    }
    // Novelty.
    const lastSeen = this.recent.lastIndexOf(m.id);
    if (lastSeen >= 0) {
      const ago = this.recent.length - lastSeen;
      score -= 4 / ago;
    }
    // Staying with one source reads as repetition; variety raises the cost.
    const sameSource = this.recent.slice(-3).filter((id) => id.split('#')[0] === m.source.id).length;
    score -= sameSource * (0.4 + 1.6 * I.variety);
    if (variant.mirrored) score += Math.log(Math.max(1e-3, I.mirror) / 0.35);
    if (variant.retro) score += Math.log(Math.max(1e-3, I.retrograde) / 0.35);
    return { score, reasons };
  }

  /** Where the dancer is: body shape (for continuity) and ground placement. */
  sense(pose) {
    if (!pose) return { here: null, place: { x: 0, z: 0, heading: 0 } };
    const sk = this.skeleton;
    const w = sk.world(pose);
    const rl = sk.jointIndex('RightLeg');
    const ll = sk.jointIndex('LeftLeg');
    const h = Math.atan2(w[rl * TF + 2] - w[ll * TF + 2], -(w[rl * TF] - w[ll * TF]));
    const rel = {};
    for (const k of Object.values(PARTS)) {
      const o = sk.jointIndex(k) * TF;
      rel[k] = [w[o] - w[0], w[o + 1], w[o + 2] - w[2]];
    }
    return { here: shapeOf(rel, h), place: { x: w[0], z: w[2], heading: h } };
  }

  /** Free choice: score every variant of every movement and sample one. */
  choose(here, exclude = null) {
    const I = this.intentions;
    const options = [];
    // "No ballet" means none: avoided material is left out (unless nothing else is known).
    const avoided = (m) => I.avoid?.some((t) => (m.tags || []).includes(t));
    let pool = this.movements.some((m) => !avoided(m)) ? this.movements.filter((m) => !avoided(m)) : this.movements;
    if (exclude) pool = pool.filter((m) => !exclude(m)).length ? pool.filter((m) => !exclude(m)) : pool;
    for (const m of pool) {
      for (const mirrored of [false, true]) {
        for (const retro of [false, true]) {
          const v = { mirrored, retro };
          const { score, reasons } = this.score(m, v, here);
          options.push({ m, v, score, reasons });
        }
      }
    }
    const temperature = 0.15 + 1.6 * I.variety;
    const max = Math.max(...options.map((o) => o.score));
    const weights = options.map((o) => Math.exp((o.score - max) / temperature));
    const total = weights.reduce((a, b) => a + b, 0);
    let r = this.random() * total;
    let pick = options.length - 1;
    for (let i = 0; i < options.length; i++) {
      r -= weights[i];
      if (r <= 0) {
        pick = i;
        break;
      }
    }
    return { ...options[pick], probability: weights[pick] / total, options: options.length };
  }

  /** The movement most like m (by its descriptors), for variations. */
  relative(m, avoid = []) {
    const keys = Object.keys(m.norm);
    let best = null;
    let bestD = Infinity;
    for (const o of this.movements) {
      if (o === m || avoid.includes(o.id)) continue;
      let d = 0;
      for (const k of keys) d += (o.norm[k] - m.norm[k]) ** 2;
      if (o.lead !== m.lead) d += 0.05;
      if (d < bestD) {
        bestD = d;
        best = o;
      }
    }
    return best || m;
  }

  /**
   * Choose and prepare the next movement. The composer decides whether this
   * step states a new motif (free choice, remembered), develops a remembered
   * one, or digresses (free choice, not remembered).
   * @param {Float64Array|null} pose  the dancer's current local pose (null at the start)
   * @returns {{ targets: object[], entry: object }}
   */
  next(pose) {
    const { here, place } = this.sense(pose);
    const step = this.composer.next(this);
    let m;
    let v;
    let reasons;
    let probability = 1;
    let options = 1;
    let score = 0;
    const extra = [];
    if (step.item) {
      m = this.movements.find((x) => x.id === step.item.id);
      v = { mirrored: step.item.mirrored, retro: step.item.retro };
      if (step.item.vary) {
        const original = m;
        m = this.relative(m, this.recent.slice(-4));
        extra.push(`in place of ${original.name}`);
      }
      reasons = [step.role];
    } else {
      const c = this.choose(here);
      ({ m, v, probability, options, score } = c);
      reasons = [step.role, ...(c.reasons.length ? c.reasons : ['chance'])];
      if (step.record) step.record({ id: m.id, mirrored: v.mirrored, retro: v.retro });
    }
    this.recent.push(m.id);
    if (this.recent.length > 24) this.recent.shift();
    const { targets, transforms } = this.realise(m, v, place, step.tempo || 1);
    const entry = {
      step: ++this.step,
      movement: m.id,
      name: m.name,
      source: m.source.name,
      credit: m.source.credit || '',
      phrase: step.phrase,
      transforms: [...transforms, ...extra],
      reasons,
      score: +score.toFixed(2),
      probability: +probability.toFixed(3),
      options,
      targets: targets.length,
    };
    this.log.push(entry);
    return { targets, entry };
  }

  /** Turn a movement variant into MotionBricks targets placed at the dancer. */
  realise(m, v, place, tempoScale = 1) {
    const I = this.intentions;
    const J = this.J;
    // Placement: the movement's first pose starts where the dancer is, facing
    // the dancer's way; near the edge, turn so its travel heads back inward.
    const transforms = [];
    if (v.mirrored) transforms.push('mirrored');
    if (v.retro) transforms.push('backwards');
    const keys = v.retro ? [...m.keys].reverse() : m.keys;
    const startHeading = v.retro ? m.endHeading : m.startHeading;
    const sample = (t, dir) => {
      const poses = new Float64Array(POSES_PER_TOKEN * J * TF);
      for (let i = 0; i < POSES_PER_TOKEN; i++) {
        const p = m.source.clip.sample(Math.min(m.source.clip.duration, Math.max(0, t + (dir * i) / SAMPLE_RATE)));
        poses.set(v.mirrored ? this.mirror(p) : p, i * J * TF);
      }
      return poses;
    };
    const dir = v.retro ? -1 : 1;
    const windows = keys.map((t) => sample(t, dir));
    const first = windows[0];
    const fh = v.mirrored ? -startHeading : startHeading;
    let yaw = place.heading - fh;
    const lastW = windows[windows.length - 1];
    const travelVec = [lastW[0] - first[0], lastW[2] - first[2]];
    const rotate = (a, [x, z]) => [Math.cos(a) * x + Math.sin(a) * z, -Math.sin(a) * x + Math.cos(a) * z];
    let end = rotate(yaw, travelVec);
    end = [place.x + end[0], place.z + end[1]];
    if (Math.hypot(end[0], end[1]) > STAGE_RADIUS) {
      // Point the travel toward the centre.
      const want = Math.atan2(-place.x, -place.z);
      const have = Math.atan2(travelVec[0], travelVec[1]);
      const newYaw = want - have;
      const deg = Math.round((((newYaw - yaw) * 180) / Math.PI + 540) % 360 - 180);
      yaw = newYaw;
      transforms.push(`turned ${deg}° to stay on stage`);
    }
    // world = T(place, yaw) * T(-first ground)
    const toOrigin = tf([-first[0], 0, -first[2]]);
    const T = tmul(tf([place.x, 0, place.z], qFromAxisAngle([0, 1, 0], yaw)), toOrigin);
    const placed = windows.map((w) => placePoses(w, J, T));

    // Tempo: token counts from the time between keys.
    const tempo = (I.tempo || 1) * tempoScale;
    if (Math.abs(tempo - 1) > 0.05) transforms.push(tempo > 1 ? `faster ×${tempo.toFixed(2)}` : `slower ×${(1 / tempo).toFixed(2)}`);
    const tokensFor = (gap) => {
      const frames = (gap * SAMPLE_RATE) / tempo;
      const t = Math.round((frames + 2 * (POSES_PER_TOKEN - 1)) / POSES_PER_TOKEN);
      const c = Math.min(MAX_TOKENS, Math.max(MIN_TOKENS, t));
      return [Math.max(MIN_TOKENS, c - 1), Math.min(MAX_TOKENS, c + 1)];
    };
    const targets = placed.map((poses, i) => {
      const range = i === 0 ? (tempo > 1.2 ? [6, 10] : tempo < 0.8 ? [10, 16] : [8, 14]) : tokensFor(Math.abs(keys[i] - keys[i - 1]));
      return { poses, numPoses: 4, tokens: rangeOf(range[0], range[1]), early: 0, movement: m.id };
    });
    return { targets, transforms };
  }
}

function rangeOf(a, b) {
  const r = [];
  for (let i = a; i <= b; i++) r.push(i);
  return r;
}
