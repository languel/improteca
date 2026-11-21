// A score: what the dancer is asked to do, in order.
//
// A prompt like "stand in a neutral pose, then walk in a circle" is a
// sequence of directed steps, not a mood. Each step says what to do:
//
//   stand   hold a natural standing pose where the dancer is (seconds, or
//           until the next prompt when it is the last step)
//   walk    walk a path: a circle (radius, direction, laps) or a straight
//           line (distance)
//   dance   improvise with the given intentions (seconds; the last dance
//           step goes on until the next prompt)
//
// The director turns each step into MotionBricks targets. It does not mix
// clips: it borrows a body pose from movement the dancer has learned (a calm
// standing frame, one phase of a walking gait), places it in space (on the
// path, facing along it) and lets MotionBricks generate the motion between.
// Steps it has no material for fall back to the improviser.
//
// A prompt with no directed step is pure improvisation, as before.

import { TF, tf, tmul, qFromAxisAngle } from '../engine/math.js';
import { SAMPLE_RATE, POSES_PER_TOKEN, MIN_TOKENS, MAX_TOKENS } from '../engine/soma.js';
import { placePoses } from './transforms.js';
import { analyse } from './repertoire.js';
import { parsePrompt } from './intentions.js';

export const ACTIONS = ['stand', 'walk', 'dance'];
export const PATHS = ['circle', 'line'];
export const DIRECTIONS = ['left', 'right'];
const STAGE_RADIUS = 3;
const HOLD = 1.2; // seconds per standing target
const DEFAULT_STAND = 4;
const DEFAULT_DANCE = 8;
const DEFAULT_RADIUS = 1.5;
const DEFAULT_DISTANCE = 3;

// ---- reading a prompt as a score (built-in, no model) -----------------------------------------

const SPLIT = /\b(?:and then|then|after that|afterwards|followed by|finally|next)\b|[;.]/;
const STAND = /\b(stand(?:s|ing)?|stay(?:ing)? still|be still|hold (?:still|a pose|the pose|it)|pose|wait(?:ing)?|rest(?:ing)?|neutral)\b/;
const WALK = /\b(walk(?:s|ing)?|stroll(?:s|ing)?|pac(?:e|ing)|go(?:es)? (?:round|around)|step(?:s|ping)? (?:forward|around))\b/;
const CIRCLE = /\b(circles?|circular|round|around|loop)\b/;
const LINE = /\b(forwards?|straight|ahead|across|line)\b/;
const NUMBER_WORDS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, ten: 10, a: 1, an: 1 };
const num = (s) => (s in NUMBER_WORDS ? NUMBER_WORDS[s] : Number(s));
const NUM = '(\\d+(?:\\.\\d+)?|one|two|three|four|five|six|ten|an?)';

function readSeconds(text) {
  const m = text.match(new RegExp(`${NUM}\\s*(?:s|secs?|seconds?)\\b`));
  if (m) return num(m[1]);
  const min = text.match(new RegExp(`${NUM}\\s*(?:mins?|minutes?)\\b`));
  if (min) return num(min[1]) * 60;
  if (/\b(a moment|a few seconds|briefly)\b/.test(text)) return 3;
  return null;
}

function readMetres(text) {
  const m = text.match(new RegExp(`${NUM}\\s*(?:m|metres?|meters?)\\b`));
  return m ? num(m[1]) : null;
}

function readLaps(text) {
  if (/\btwice\b/.test(text)) return 2;
  if (/\bonce\b/.test(text)) return 1;
  const m = text.match(new RegExp(`${NUM}\\s*(?:times|laps?|circles)\\b`));
  return m ? num(m[1]) : null;
}

function readDirection(text) {
  if (/\b(counter-?clockwise|anti-?clockwise)\b/.test(text)) return 'left';
  if (/\bclockwise\b/.test(text)) return 'right';
  if (/\b(to the |towards? the |on the )?left\b/.test(text)) return 'left';
  if (/\b(to the |towards? the |on the )?right\b/.test(text)) return 'right';
  return null;
}

/** One clause as a step (or null when it is not a directed action). */
function readStep(clause) {
  if (WALK.test(clause)) {
    const circle = CIRCLE.test(clause) || !LINE.test(clause) && /\bin a\b.*\b(ring|round)\b/.test(clause);
    if (circle) {
      let radius = readMetres(clause);
      if (radius === null) radius = /\b(small|tight|little)\b/.test(clause) ? 1 : /\b(big|large|wide)\b/.test(clause) ? 2.5 : null;
      return { action: 'walk', path: 'circle', radius, direction: readDirection(clause), laps: readLaps(clause), distance: null, seconds: null };
    }
    return { action: 'walk', path: 'line', distance: readMetres(clause), radius: null, direction: null, laps: null, seconds: null };
  }
  if (STAND.test(clause)) return { action: 'stand', seconds: readSeconds(clause), path: null, radius: null, distance: null, direction: null, laps: null };
  return null;
}

/**
 * Read a prompt as a score with the built-in vocabulary.
 * @returns {{ steps: object[], understood: string[] }} steps is [] when the
 *   prompt asks for no directed action (pure improvisation).
 */
export function parseScore(prompt) {
  const text = ` ${String(prompt).toLowerCase().replace(/\s+/g, ' ')} `;
  const clauses = text.split(SPLIT).map((c) => c.replace(/^\s*(first|to start|start by|begin by)\b/, '').trim()).filter(Boolean);
  const steps = [];
  let directed = false;
  for (const clause of clauses) {
    const step = readStep(clause);
    if (step) {
      directed = true;
      steps.push(step);
    } else {
      const { intentions } = parsePrompt(clause);
      steps.push({ action: 'dance', seconds: readSeconds(clause), intentions, path: null, radius: null, distance: null, direction: null, laps: null });
    }
  }
  if (!directed) return { steps: [], understood: [] };
  const out = sanitizeSteps(steps);
  return { steps: out, understood: out.map(describeStep) };
}

/** Whatever a model returned, as valid steps (unknown actions dropped, numbers clamped). */
export function sanitizeSteps(raw) {
  if (!Array.isArray(raw)) return [];
  const n = (v, lo, hi) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : null);
  return raw.slice(0, 12).filter((s) => ACTIONS.includes(s?.action)).map((s) => ({
    action: s.action,
    seconds: n(s.seconds, 0.5, 600),
    path: s.action === 'walk' ? (PATHS.includes(s.path) ? s.path : 'line') : null,
    radius: n(s.radius, 0.6, STAGE_RADIUS),
    distance: n(s.distance, 0.5, 2 * STAGE_RADIUS),
    direction: DIRECTIONS.includes(s.direction) ? s.direction : null,
    laps: n(s.laps, 0.25, 4),
    ...(s.intentions ? { intentions: s.intentions } : {}),
  }));
}

/** A short label for a step. */
export function describeStep(s) {
  const secs = s.seconds ? ` ${+s.seconds.toFixed(1)} s` : '';
  if (s.action === 'stand') return `stand${secs}`;
  if (s.action === 'walk') {
    if (s.path === 'circle') {
      const laps = s.laps && s.laps !== 1 ? ` ×${+s.laps.toFixed(2)}` : '';
      return `walk a circle${s.direction ? ` to the ${s.direction}` : ''} (r ${+(s.radius ?? DEFAULT_RADIUS).toFixed(1)} m)${laps}`;
    }
    return `walk ${+(s.distance ?? DEFAULT_DISTANCE).toFixed(1)} m`;
  }
  return `dance${secs}`;
}

// ---- material: what to borrow a body pose from -----------------------------------------------

const analysisOf = (source, skeleton) => (source.analysis ??= analyse(source.clip, skeleton));

/** Horizontal speed of the hips per frame (m/s), centred over half a second. */
function groundSpeed(frames) {
  const w = 7;
  return frames.map((_, f) => {
    const a = frames[Math.max(0, f - w)].root;
    const b = frames[Math.min(frames.length - 1, f + w)].root;
    const dt = (Math.min(frames.length - 1, f + w) - Math.max(0, f - w)) / SAMPLE_RATE;
    return dt > 0 ? Math.hypot(b[0] - a[0], b[2] - a[2]) / dt : 0;
  });
}

/** Longest run of frames where ok(f) holds: [start, end] (inclusive) or null. */
function longestRun(n, ok) {
  let best = null;
  let start = -1;
  for (let f = 0; f <= n; f++) {
    if (f < n && ok(f)) {
      if (start < 0) start = f;
    } else if (start >= 0) {
      if (!best || f - 1 - start > best[1] - best[0]) best = [start, f - 1];
      start = -1;
    }
  }
  return best;
}

/** The calmest stretch of standing in the repertoire: { source, start, end } in seconds. */
export function findStanding(sources, skeleton) {
  let best = null;
  for (const source of sources) {
    const a = analysisOf(source, skeleton);
    const v = groundSpeed(a.frames);
    // Neutral standing: slow, upright, feet on the floor under the body, arms hanging.
    const neutral = (f) => {
      const fr = a.frames[f];
      const r = fr.rel;
      const below = (h) => r[h][1] < fr.root[1] + 0.1 && Math.hypot(r[h][0], r[h][2]) < 0.4;
      return a.speed[f] < 0.3 && v[f] < 0.4 && fr.root[1] / a.restHips > 0.94 && fr.toeY < 0.08 && below('LeftHand') && below('RightHand')
        && Math.hypot(r.LeftFoot[0] - r.RightFoot[0], r.LeftFoot[2] - r.RightFoot[2]) < 0.5;
    };
    const run = longestRun(a.frames.length, neutral);
    if (!run) continue;
    const length = (run[1] - run[0]) / SAMPLE_RATE;
    if (length < 1) continue;
    const score = Math.min(length, 6) + ((source.tags || []).includes('everyday') ? 3 : 0);
    if (!best || score > best.score) best = { source, start: run[0] / SAMPLE_RATE, end: run[1] / SAMPLE_RATE, score };
  }
  return best;
}

/**
 * A steady stretch of walking: { source, time, heading, speed, period }.
 * time is one moment of the gait whose four-pose window all walk targets
 * share; heading is the direction of travel there; period is one gait cycle
 * (two steps) in seconds.
 */
export function findWalk(sources, skeleton) {
  let best = null;
  for (const source of sources) {
    const a = analysisOf(source, skeleton);
    const fr = a.frames;
    const v = groundSpeed(fr);
    const turnRate = fr.map((_, f) => {
      const p = fr[Math.max(0, f - 7)].root;
      const q = fr[f].root;
      const r = fr[Math.min(fr.length - 1, f + 7)].root;
      const h1 = Math.atan2(q[0] - p[0], q[2] - p[2]);
      const h2 = Math.atan2(r[0] - q[0], r[2] - q[2]);
      return Math.abs(Math.atan2(Math.sin(h2 - h1), Math.cos(h2 - h1))) * (SAMPLE_RATE / 7);
    });
    const run = longestRun(fr.length, (f) => v[f] > 0.7 && v[f] < 1.9 && turnRate[f] < 0.6 && fr[f].toeY < 0.15);
    if (!run || run[1] - run[0] < 1.2 * SAMPLE_RATE) continue;
    // Gait period: the lag (0.8 to 1.6 s) at which the left-right foot height difference repeats best.
    const s = fr.map((x) => x.rel.LeftFoot[1] - x.rel.RightFoot[1]);
    let period = 1.1;
    let bestCorr = -Infinity;
    for (let lag = Math.round(0.8 * SAMPLE_RATE); lag <= Math.round(1.6 * SAMPLE_RATE); lag++) {
      if (run[0] + lag + 10 > run[1]) break;
      let c = 0;
      let n = 0;
      for (let f = run[0]; f + lag <= run[1]; f++) {
        c += s[f] * s[f + lag];
        n++;
      }
      c /= n;
      if (c > bestCorr) {
        bestCorr = c;
        period = lag / SAMPLE_RATE;
      }
    }
    // The shared gait moment: feet level (double support), early in the run, with room for the window.
    const lo = run[0] + 6;
    const hi = Math.min(run[1] - POSES_PER_TOKEN - 7, lo + Math.round(period * SAMPLE_RATE));
    if (hi < lo) continue;
    let f0 = lo;
    for (let f = lo; f <= hi; f++) if (Math.abs(s[f]) < Math.abs(s[f0])) f0 = f;
    const p = fr[Math.max(run[0], f0 - 7)].root;
    const q = fr[Math.min(run[1], f0 + 7)].root;
    const speed = v.slice(run[0], run[1] + 1).reduce((x, y) => x + y, 0) / (run[1] - run[0] + 1);
    const cand = { source, time: f0 / SAMPLE_RATE, heading: Math.atan2(q[0] - p[0], q[2] - p[2]), speed, period, length: (run[1] - run[0]) / SAMPLE_RATE };
    if (!best || cand.length > best.length) best = cand;
  }
  return best;
}

// ---- the director ---------------------------------------------------------------------------

const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const forward = (h) => [Math.sin(h), Math.cos(h)];

function tokensFor(seconds) {
  const frames = seconds * SAMPLE_RATE;
  const t = Math.round((frames + 2 * (POSES_PER_TOKEN - 1)) / POSES_PER_TOKEN);
  const c = Math.min(MAX_TOKENS, Math.max(MIN_TOKENS, t));
  const out = [];
  for (let i = Math.max(MIN_TOKENS, c - 1); i <= Math.min(MAX_TOKENS, c + 1); i++) out.push(i);
  return out;
}

/** Points along a path: { x, z, heading } at arc lengths s (heading = direction of travel). */
export function pathPoint(start, step, s) {
  if (step.path === 'circle') {
    const r = step.radius ?? DEFAULT_RADIUS;
    const k = step.direction === 'right' ? -1 : 1; // left: heading increases
    const h = start.heading + (k * s) / r;
    return {
      x: start.x + k * r * (Math.cos(start.heading) - Math.cos(h)),
      z: start.z + k * r * (Math.sin(h) - Math.sin(start.heading)),
      heading: h,
    };
  }
  const [fx, fz] = forward(start.heading);
  return { x: start.x + fx * s, z: start.z + fz * s, heading: start.heading };
}

export function pathLength(step) {
  if (step.path === 'circle') return 2 * Math.PI * (step.radius ?? DEFAULT_RADIUS) * (step.laps ?? 1);
  return step.distance ?? DEFAULT_DISTANCE;
}

export class Director {
  /**
   * @param {{ improviser: import('./improviser.js').Improviser, skeleton: object }} opts
   */
  constructor({ improviser, skeleton }) {
    this.improviser = improviser;
    this.skeleton = skeleton;
    this.J = skeleton.numJoints;
    this.steps = [];
    this.intentions = null;
    this.index = 0;
    this.planned = 0; // seconds planned in the current step
    this.announced = false;
    this.standing = null;
    this.walking = null;
    this.materialFrom = 0;
  }

  /** A new score (steps may be []: pure improvisation with the intentions). */
  setScore(steps, intentions) {
    this.steps = steps || [];
    this.intentions = intentions;
    this.index = 0;
    this.planned = 0;
    this.announced = false;
    this.improviser.setIntentions(this.steps[0]?.intentions ?? intentions);
  }

  get sources() {
    return [...new Set(this.improviser.movements.map((m) => m.source))];
  }

  /** Find standing and walking material again when the repertoire has grown. */
  material() {
    const n = this.improviser.movements.length;
    if (n !== this.materialFrom) {
      this.materialFrom = n;
      const sources = this.sources;
      this.standing = findStanding(sources, this.skeleton);
      this.walking = findWalk(sources, this.skeleton);
    }
    return { standing: this.standing, walking: this.walking };
  }

  /** The step now (or the implied last one: hold still after a directed score). */
  current() {
    while (this.index < this.steps.length) {
      const s = this.steps[this.index];
      const last = this.index === this.steps.length - 1;
      const limit = s.action === 'walk' ? (s.done ? 0 : Infinity) : last ? Infinity : (s.seconds ?? (s.action === 'stand' ? DEFAULT_STAND : DEFAULT_DANCE));
      if (this.planned < limit) return { step: s, last };
      this.index++;
      this.planned = 0;
      this.announced = false;
      const next = this.steps[this.index];
      if (next) this.improviser.setIntentions(next.intentions ?? this.intentions);
    }
    // After a walk as the last step: stand where the path ended.
    return { step: { action: 'stand', seconds: null, implied: true }, last: true };
  }

  /**
   * The next targets, the same contract as Improviser.next.
   * @param {Float64Array|null} pose  the last generated pose (null at the start)
   */
  next(pose) {
    if (!this.steps.length) return this.improviser.next(pose);
    const { place } = this.improviser.sense(pose);
    if (!pose) {
      // Begin from standing, whatever comes first.
      const stand = this.stand(place, 1);
      if (stand) return stand;
    }
    const { step, last } = this.current();
    const phrase = `score ${Math.min(this.index + 1, this.steps.length)}/${this.steps.length}`;
    if (step.action === 'stand') {
      const out = this.stand(place, last ? 3 : Math.max(1, Math.ceil(((step.seconds ?? DEFAULT_STAND) - this.planned) / HOLD)), phrase, pose);
      if (out) return out;
    } else if (step.action === 'walk') {
      const out = this.walk(place, step, phrase);
      if (out) return out;
      // No walking material: travel with the improviser instead.
      step.done = true;
      this.improviser.setIntentions({ ...this.intentions, travel: 0.95, jump: 0, turn: 0.2 });
    }
    const out = this.improviser.next(pose);
    this.planned += out.targets.length * 1.1;
    out.entry.phrase = `${phrase} · ${out.entry.phrase || ''}`.replace(/ · $/, '');
    return out;
  }

  entry(name, source, reasons, transforms, phrase, targets) {
    const e = {
      step: ++this.improviser.step,
      movement: name,
      name,
      source: source?.name || '',
      credit: source?.credit || '',
      phrase,
      transforms,
      reasons,
      score: 0,
      probability: 1,
      options: 1,
      targets,
    };
    this.improviser.log.push(e);
    return e;
  }

  /** Four poses from a source as recorded, starting at t. */
  window(source, t) {
    const J = this.J;
    const poses = new Float64Array(POSES_PER_TOKEN * J * TF);
    const clip = source.clip;
    for (let i = 0; i < POSES_PER_TOKEN; i++) poses.set(clip.sample(Math.min(clip.duration, t + i / SAMPLE_RATE)), i * J * TF);
    return poses;
  }

  /** Place a window so its hips' ground point lands at (x, z) and its heading becomes `heading`. */
  place(poses, from, x, z, heading) {
    const T = tmul(tf([x, 0, z], qFromAxisAngle([0, 1, 0], wrapAngle(heading - from.heading))), tf([-from.x, 0, -from.z]));
    return placePoses(poses, this.J, T);
  }

  /** Hold a natural standing pose at the dancer's place: `count` targets. */
  stand(place, count, phrase = 'start', pose = null) {
    const { standing } = this.material();
    let windows;
    let source = null;
    // One spot per stand step: later holds stay where the first one was placed.
    if (!this.announced || !this.anchor) this.anchor = { ...place };
    place = this.anchor;
    if (standing) {
      source = standing.source;
      const a = analysisOf(source, this.skeleton);
      const fr = a.frames[Math.round(standing.start * SAMPLE_RATE)];
      const from = { x: fr.root[0], z: fr.root[2], heading: fr.heading };
      const span = Math.max(0.01, standing.end - standing.start - POSES_PER_TOKEN / SAMPLE_RATE);
      windows = [];
      for (let i = 0; i < count; i++) {
        // Walk through the standing stretch, so the hold breathes and shifts weight as recorded.
        const t = standing.start + ((((this.planned + i * HOLD) % span) + span) % span);
        windows.push(this.place(this.window(source, t), from, place.x, place.z, place.heading));
      }
    } else if (pose) {
      // Nothing learned to stand with: hold the current pose.
      const one = new Float64Array(POSES_PER_TOKEN * this.J * TF);
      for (let i = 0; i < POSES_PER_TOKEN; i++) one.set(pose, i * this.J * TF);
      windows = Array.from({ length: count }, () => one);
    } else return null;
    this.planned += count * HOLD;
    const targets = windows.map((poses) => ({ poses, numPoses: POSES_PER_TOKEN, tokens: tokensFor(HOLD), early: 0, movement: 'stand' }));
    let entry = null;
    if (!this.announced) {
      this.announced = true;
      entry = this.entry('Standing', source, ['as asked'], source ? ['held in place'] : ['holding the current pose'], phrase, targets.length);
    }
    return { targets, entry };
  }

  /** Walk a path from the dancer's place: one target per gait cycle along it. */
  walk(place, step, phrase) {
    const { walking } = this.material();
    if (!walking) return null;
    const s = { ...step };
    if (s.path === 'circle' && !s.direction) {
      // Turn toward the middle of the stage, so the circle stays on it.
      const [lx, lz] = [Math.cos(place.heading), -Math.sin(place.heading)]; // the dancer's left
      s.direction = lx * -place.x + lz * -place.z >= 0 ? 'left' : 'right';
    }
    let start = place;
    if (s.path === 'line') {
      const end = pathPoint(place, s, pathLength(s));
      if (Math.hypot(end.x, end.z) > STAGE_RADIUS + 0.5) start = { ...place, heading: Math.atan2(-place.x, -place.z) };
    }
    const length = pathLength(s);
    const stride = walking.speed * walking.period;
    const n = Math.max(1, Math.round(length / stride));
    const ds = length / n;
    const seconds = ds / walking.speed;
    const window = this.window(walking.source, walking.time);
    const a = analysisOf(walking.source, this.skeleton);
    const fr = a.frames[Math.round(walking.time * SAMPLE_RATE)];
    const from = { x: fr.root[0], z: fr.root[2], heading: walking.heading };
    const targets = [];
    for (let k = 1; k <= n; k++) {
      const p = pathPoint(start, s, k * ds);
      // The first cycle starts from rest: give it a little longer.
      const tokens = tokensFor(k === 1 ? seconds * 1.3 : seconds);
      targets.push({ poses: this.place(window, from, p.x, p.z, p.heading), numPoses: POSES_PER_TOKEN, tokens, early: 0, movement: 'walk' });
    }
    step.done = true;
    this.planned += n * seconds;
    const transforms = [`${n} gait cycles of ${ds.toFixed(2)} m`];
    if (start !== place) transforms.push('turned toward the centre to stay on stage');
    const entry = this.entry(`Walking: ${describeStep(s).replace(/^walk /, '')}`, walking.source, ['as asked'], transforms, phrase, targets.length);
    return { targets, entry };
  }
}
