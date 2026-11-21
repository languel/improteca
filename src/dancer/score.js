// A score: what the dancer is asked to do, in order.
//
// A prompt like "stand in a neutral pose, then walk in a circle" is a
// sequence of directed steps, not a mood. Each step says what to do:
//
//   stand   hold a natural standing pose where the dancer is (seconds, or
//           until the next prompt when it is the last step)
//   walk    walk a floor pattern, written with a small general toolkit
//           (turtle geometry), so any shape can be composed rather than
//           picked from a list:
//             forward  distance in metres along the way the dancer faces
//             turn     angle in degrees, on the spot (positive: to the left)
//             arc      radius and angle (positive: curving to the left)
//           and `repeat` for the whole list. A square is forward 2, turn 90,
//           repeated 4 times; a figure of eight is arc +360 then arc -360.
//   jump    jump (improvised from learned airborne movement), for a few
//           seconds
//   dance   improvise with the given intentions (seconds; the last dance
//           step goes on until the next prompt)
//
// Working out a shape is the brain's job (see brains.js); the built-in
// vocabulary only reads plain circles, distances and turns.
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

export const ACTIONS = ['stand', 'walk', 'jump', 'dance'];
export const MOVES = ['forward', 'turn', 'arc'];
const STAGE_RADIUS = 3;
const HOLD = 1.8; // seconds per standing target: long spans, so MotionBricks is not hurried
const SHIFT_EVERY = 3; // holds between weight shifts (about 5 s)
const DEFAULT_STAND = 4;
const DEFAULT_DANCE = 8;
const DEFAULT_JUMP = 3;
const DEFAULT_RADIUS = 1.5;
const DEFAULT_DISTANCE = 3;
const MAX_TARGETS = 80; // per walk, about a minute and a half of walking
const DEG = Math.PI / 180;
const KIMODO_MAX = 10; // seconds per Kimodo clip
const KIMODO_GAIT = { speed: 1.1, period: 1.1 }; // where walk waypoints go when Kimodo walks
// What to ask Kimodo for when the brain gave no description of its own.
const KIMODO_TEXT = {
  stand: 'A person stands calmly in a relaxed, natural pose, shifting their weight now and then.',
  walk: 'A person walks at a relaxed, natural pace.',
  jump: 'A person jumps up energetically and lands softly.',
};

// ---- reading a prompt as a score (built-in, no model) -----------------------------------------

const SPLIT = /\b(?:and then|then|after that|afterwards|followed by|finally|next)\b|[;.]/;
const STAND = /\b(stand(?:s|ing)?|stay(?:ing)? still|be still|hold (?:still|a pose|the pose|it)|pose|wait(?:ing)?|rest(?:ing)?|neutral)\b/;
const WALK = /\b(walk(?:s|ing)?|stroll(?:s|ing)?|pac(?:e|ing)|go(?:es)? (?:round|around)|step(?:s|ping)? (?:forward|around))\b/;
const CIRCLE = /\b(circles?|circular|round|around|loop)\b/;
const TURN = /^(?:turn|rotate|spin|face)\b/;
// A jump as an action: the clause starts with it ("jump", "then jump twice"), not "ballet with jumps".
const JUMP = /^(?:do (?:a |some )?|make (?:a |some )?)?(jump|hop|leap)(?:s|ing)?\b/;
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

function readTimes(text) {
  if (/\btwice\b/.test(text)) return 2;
  if (/\bonce\b/.test(text)) return 1;
  const m = text.match(new RegExp(`${NUM}\\s*(?:times|laps?|circles)\\b`));
  return m ? num(m[1]) : null;
}

/** +1 for left (counterclockwise seen from above), -1 for right, null if unsaid. */
function readSide(text) {
  if (/\b(counter-?clockwise|anti-?clockwise)\b/.test(text)) return 1;
  if (/\bclockwise\b/.test(text)) return -1;
  if (/\bleft\b/.test(text)) return 1;
  if (/\bright\b/.test(text)) return -1;
  return null;
}

const step = (action, fields = {}) => ({ action, seconds: null, moves: null, repeat: null, ...fields });

/** One clause as a step (or null when it is not a directed action). */
function readStep(clause) {
  if (WALK.test(clause)) {
    if (CIRCLE.test(clause)) {
      let radius = readMetres(clause);
      if (radius === null) radius = /\b(small|tight|little)\b/.test(clause) ? 1 : /\b(big|large|wide)\b/.test(clause) ? 2.5 : DEFAULT_RADIUS;
      const side = readSide(clause);
      // No side given: the director picks the side that keeps it on stage.
      return step('walk', { moves: [{ move: 'arc', radius, angle: 360 * (side ?? 1), distance: null, free: side === null }], repeat: readTimes(clause) });
    }
    return step('walk', { moves: [{ move: 'forward', distance: readMetres(clause) ?? DEFAULT_DISTANCE, angle: null, radius: null }] });
  }
  if (TURN.test(clause)) {
    const m = clause.match(/(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?)/);
    const angle = m ? Number(m[1]) : /\b(around|about)\b/.test(clause) ? 180 : 90;
    return step('walk', { moves: [{ move: 'turn', angle: angle * (readSide(clause) ?? 1), distance: null, radius: null }] });
  }
  if (JUMP.test(clause)) {
    const times = readTimes(clause);
    return step('jump', { seconds: readSeconds(clause) ?? (times ? times * 1.5 : null) });
  }
  if (STAND.test(clause)) return step('stand', { seconds: readSeconds(clause) });
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
    const s = readStep(clause);
    if (s) {
      directed = true;
      steps.push(s);
    } else {
      steps.push(step('dance', { seconds: readSeconds(clause), intentions: parsePrompt(clause).intentions }));
    }
  }
  if (!directed) return { steps: [], understood: [] };
  const out = sanitizeSteps(steps);
  return { steps: out, understood: out.map(describeStep) };
}

/** Whatever a model returned, as valid steps (unknown actions and moves dropped, numbers clamped). */
export function sanitizeSteps(raw) {
  if (!Array.isArray(raw)) return [];
  const n = (v, lo, hi) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : null);
  const move = (m) => {
    if (m?.move === 'forward') return { move: 'forward', distance: n(m.distance, 0.1, 2 * STAGE_RADIUS) ?? 1, angle: null, radius: null };
    if (m?.move === 'turn') return { move: 'turn', angle: n(m.angle, -1440, 1440) ?? 90, distance: null, radius: null };
    if (m?.move === 'arc') return { move: 'arc', radius: n(m.radius, 0.3, STAGE_RADIUS) ?? DEFAULT_RADIUS, angle: n(m.angle, -1440, 1440) ?? 360, distance: null, ...(m.free ? { free: true } : {}) };
    return null;
  };
  return raw.slice(0, 12).filter((s) => ACTIONS.includes(s?.action)).map((s) => {
    const out = { action: s.action, seconds: n(s.seconds, 0.5, 600), moves: null, repeat: null };
    if (typeof s.text === 'string' && s.text.trim()) out.text = s.text.trim().slice(0, 300);
    if (s.action === 'walk') {
      // Moves that do nothing (an arc of 0 degrees) are dropped.
      out.moves = (Array.isArray(s.moves) ? s.moves.slice(0, 24).map(move).filter((m) => m && (m.move !== 'arc' || Math.abs(m.angle) >= 1) && (m.move !== 'turn' || m.angle !== 0)) : []);
      if (typeof s.plan === 'string' && s.plan.trim()) out.plan = s.plan.trim().slice(0, 200);
      if (!out.moves.length) out.moves = [{ move: 'forward', distance: DEFAULT_DISTANCE, angle: null, radius: null }];
      out.repeat = Math.round(n(s.repeat, 1, 12) ?? 1);
    }
    if (s.intentions) out.intentions = s.intentions;
    return out;
  });
}

/**
 * Combine a model's steps with the built-in reading of the same prompt: the
 * model's steps win (they carry its geometry and qualities), and an action
 * the built-in reading found that the model left out (a small model may drop
 * "then jump") is put back in its place in the sequence.
 * @returns {{ steps: object[], added: object[] }}
 */
export function mergeSteps(model, builtIn) {
  if (!model.length) return { steps: builtIn.slice(), added: builtIn.slice() };
  const steps = [];
  const added = [];
  let j = 0;
  for (let i = 0; i < builtIn.length; i++) {
    const b = builtIn[i];
    const at = model.findIndex((m, k) => k >= j && m.action === b.action);
    if (at >= 0) {
      steps.push(...model.slice(j, at + 1));
      j = at + 1;
    } else {
      steps.push(b);
      added.push(b);
    }
  }
  steps.push(...model.slice(j));
  return { steps, added };
}

const fmt = (x) => +x.toFixed(1);

/** A short label for a step. */
export function describeStep(s) {
  const secs = s.seconds ? ` ${fmt(s.seconds)} s` : '';
  if (s.action === 'stand') return `stand${secs}`;
  if (s.action === 'jump') return `jump${secs}`;
  if (s.action === 'walk') {
    const side = (a) => (a >= 0 ? 'left' : 'right');
    const parts = (s.moves || []).map((m) => (m.move === 'forward' ? `forward ${fmt(m.distance)} m`
      : m.move === 'turn' ? `turn ${side(m.angle)} ${fmt(Math.abs(m.angle))}°`
        : `arc ${m.free ? '' : `${side(m.angle)} `}${fmt(Math.abs(m.angle))}° r ${fmt(m.radius)} m`));
    return `walk: ${parts.join(', ')}${s.repeat > 1 ? `, ×${s.repeat}` : ''}`;
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

/**
 * The calmest stretch of standing in the repertoire: { source, start, end,
 * rest, shift } in seconds. rest is its calmest moment; shift is a calm
 * moment with the weight more on the other leg, for an occasional shift.
 */
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
  if (!best) return null;
  // Calm moments: the slowest third of the stretch, with room for a four-pose window.
  const a = analysisOf(best.source, skeleton);
  const frames = [];
  for (let f = Math.round(best.start * SAMPLE_RATE); f <= Math.round(best.end * SAMPLE_RATE) - POSES_PER_TOKEN; f++) frames.push(f);
  const calm = frames.slice().sort((x, y) => a.speed[x] - a.speed[y]).slice(0, Math.max(1, Math.ceil(frames.length / 3)));
  // Where the hips are over the feet, sideways (shape x is the body's side axis).
  const over = (f) => -(a.frames[f].shape[6] + a.frames[f].shape[9]) / 2;
  const rest = calm[0];
  let shift = rest;
  for (const f of calm) if (Math.abs(over(f) - over(rest)) > Math.abs(over(shift) - over(rest))) shift = f;
  best.rest = rest / SAMPLE_RATE;
  best.shift = Math.abs(over(shift) - over(rest)) > 0.015 ? shift / SAMPLE_RATE : best.rest;
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

/**
 * Movements where the dancer leaves the floor: the toes rise well above the
 * clip's own floor level and the hips rise above their usual height (so a
 * relevé or a raised leg is not a jump). Returns [{ movement, lift, rise }], highest first.
 */
export function findJumps(movements, skeleton) {
  const out = [];
  const floors = new Map();
  for (const m of movements) {
    const a = analysisOf(m.source, skeleton);
    if (!floors.has(m.source)) {
      const toes = a.frames.map((f) => f.toeY).sort((x, y) => x - y);
      const hips = a.frames.map((f) => f.root[1]).sort((x, y) => x - y);
      floors.set(m.source, { floor: toes[Math.floor(toes.length * 0.1)], hips: hips[Math.floor(hips.length / 2)] });
    }
    const { floor, hips } = floors.get(m.source);
    const span = a.frames.slice(Math.round(m.start * SAMPLE_RATE), Math.round(m.end * SAMPLE_RATE) + 1);
    const lift = Math.max(...span.map((f) => f.toeY - floor));
    let peak = 0;
    span.forEach((f, i) => (f.root[1] > span[peak].root[1] ? (peak = i) : 0));
    const rise = span[peak].root[1] - hips;
    if (lift > 0.2 && rise > 0.12) out.push({ movement: m, lift, rise, peak: m.start + peak / SAMPLE_RATE });
  }
  return out.sort((x, y) => y.lift + y.rise - (x.lift + x.rise));
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

/**
 * Where walk targets go, following the moves like a turtle:
 * [{ x, z, heading, seconds }] (seconds since the previous target). Straight
 * stretches and arcs get one target per gait cycle; a turn on the spot is
 * taken on the first step after it (a corner), or as a pivot on the spot when
 * it is large or comes last.
 */
export function waypoints(start, step, walking) {
  const stride = walking.speed * walking.period;
  const cycle = (d) => d / walking.speed;
  const out = [];
  let { x, z, heading: h } = start;
  let turned = 0; // degrees turned since the last target
  const pivot = () => {
    out.push({ x, z, heading: h, seconds: 0.6 + Math.min(1.2, Math.abs(turned) / 150) });
    turned = 0;
  };
  const moves = [];
  for (let r = 0; r < (step.repeat ?? 1); r++) moves.push(...step.moves);
  for (const m of moves) {
    if (out.length >= MAX_TARGETS) break;
    if (m.move === 'turn') {
      h += m.angle * DEG;
      turned += m.angle;
      continue;
    }
    if (Math.abs(turned) > 100) pivot();
    const length = m.move === 'arc' ? Math.abs(m.radius * m.angle * DEG) : m.distance;
    const n = Math.max(1, Math.round(length / stride));
    const ds = length / n;
    const k = m.move === 'arc' ? Math.sign(m.angle) : 0;
    for (let j = 1; j <= n && out.length < MAX_TARGETS; j++) {
      // Starting from rest, or out of a corner, takes a little longer.
      const slow = out.length === 0 ? 1.3 : j === 1 && turned ? 1 + Math.min(0.5, Math.abs(turned) / 180) : 1;
      if (k) {
        const h1 = h + (k * ds) / m.radius;
        x += k * m.radius * (Math.cos(h) - Math.cos(h1));
        z += k * m.radius * (Math.sin(h1) - Math.sin(h));
        h = h1;
      } else {
        x += Math.sin(h) * ds;
        z += Math.cos(h) * ds;
      }
      out.push({ x, z, heading: h, seconds: cycle(ds) * slow });
      turned = 0;
    }
  }
  if (turned && out.length < MAX_TARGETS) pivot();
  return out;
}

/**
 * Rotate the whole pattern about its start, as little as possible, so it
 * stays on stage. A circle whose side was not asked for may also curve the
 * other way. Returns { points, rotation (degrees), mirrored }.
 */
export function fitOnStage(start, step, walking) {
  const variants = [step];
  if (step.moves.some((m) => m.free)) variants.push({ ...step, moves: step.moves.map((m) => (m.free ? { ...m, angle: -m.angle } : m)) });
  let best = null;
  for (const deg of [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]) {
    for (const [i, v] of variants.entries()) {
      const points = waypoints({ ...start, heading: start.heading + deg * DEG }, v, walking);
      const reach = Math.max(0, ...points.map((p) => Math.hypot(p.x, p.z)));
      const cost = (reach <= STAGE_RADIUS + 0.3 ? 0 : 100 + reach) + Math.abs(deg) / 30 + i * 0.5;
      if (!best || cost < best.cost) best = { points, rotation: deg, mirrored: i === 1, cost };
    }
  }
  return best;
}

export class Director {
  /**
   * @param {{ improviser: import('./improviser.js').Improviser, skeleton: object, kimodo?: import('./kimodo.js').KimodoClient }} opts
   */
  constructor({ improviser, skeleton, kimodo = null }) {
    this.improviser = improviser;
    this.kimodo = kimodo; // text-to-motion for every step, when its server is running
    this.prompt = '';
    this.onKimodo = null; // (event: { state, step, text, ms?, error? }) => void, for the page
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

  /**
   * A new score (steps may be []: pure improvisation with the intentions).
   * With Kimodo, every step is generated from text: a prompt without steps
   * becomes one open-ended dance, and a score ending in a walk or a jump
   * ends standing.
   */
  setScore(steps, intentions, { prompt = '', summary = '' } = {}) {
    this.steps = (steps || []).map((s) => ({ ...s }));
    this.intentions = intentions;
    this.prompt = prompt;
    this.index = 0;
    this.planned = 0;
    this.announced = false;
    this.run = (this.run || 0) + 1;
    if (this.kimodo?.ready) {
      if (!this.steps.length) this.steps.push({ action: 'dance', seconds: null, moves: null, repeat: null, text: null, free: true, summary });
      if (['walk', 'jump'].includes(this.steps[this.steps.length - 1].action)) {
        this.steps.push({ action: 'stand', seconds: null, moves: null, repeat: null, text: null, implied: true });
      }
      this.steps.forEach((step, i) => {
        step.k = { queue: [], pending: null, failed: false, finished: false, chunks: null, next: 0, seed: (this.improviser.seed * 7919 + i * 101) % 100000 };
      });
      // Ask for every step now (the server takes them in order), so later steps are ready in time.
      this.steps.forEach((step, i) => this.request(step, i));
    }
    this.improviser.setIntentions(this.intentionsFor(this.steps[0]));
  }

  /** What the improviser works with during a step. */
  intentionsFor(step) {
    const base = step?.intentions ?? this.intentions ?? {};
    if (step?.action === 'jump') return { ...base, jump: 1, energy: Math.max(base.energy ?? 0, 0.75), stillness: 0, travel: null, variety: 0.3, tempo: 1 };
    return base;
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
      this.jumps = findJumps(this.improviser.movements, this.skeleton);
    }
    return { standing: this.standing, walking: this.walking, jumps: this.jumps };
  }

  /** The step now (or the implied last one: hold still after a directed score). */
  current() {
    while (this.index < this.steps.length) {
      const s = this.steps[this.index];
      const last = this.index === this.steps.length - 1;
      const limit = s.k && !s.k.failed ? (s.k.finished ? 0 : Infinity)
        : s.action === 'walk' ? (s.done ? 0 : Infinity)
        : s.action === 'jump' ? (s.seconds ?? DEFAULT_JUMP)
          : last ? Infinity : (s.seconds ?? (s.action === 'stand' ? DEFAULT_STAND : DEFAULT_DANCE));
      if (this.planned < limit) return { step: s, last };
      this.index++;
      this.planned = 0;
      this.announced = false;
      const next = this.steps[this.index];
      if (next) this.improviser.setIntentions(this.intentionsFor(next));
    }
    // After a walk or a jump as the last step: stand where it ended.
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
    if (step.k && !step.k.failed) {
      const out = this.fromKimodo(step, last, place, pose, phrase);
      if (out) return out;
    }
    if (step.action === 'stand') {
      const out = this.stand(place, last ? 3 : Math.max(1, Math.ceil(((step.seconds ?? DEFAULT_STAND) - this.planned) / HOLD)), phrase, pose);
      if (out) return out;
    } else if (step.action === 'jump') {
      const out = this.jump(place, phrase);
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

  // ---- Kimodo -----------------------------------------------------------------------------

  /** What to ask Kimodo for, for one step: [{ text, seconds, path? }] (a walk longer than a clip is split). */
  kimodoChunks(step, index) {
    const last = index === this.steps.length - 1;
    const text = step.text || KIMODO_TEXT[step.action] || `A person dances. ${step.summary || this.prompt}`.trim();
    if (step.action === 'walk') {
      const points = waypoints({ x: 0, z: 0, heading: 0 }, step, KIMODO_GAIT);
      const chunks = [];
      let from = { x: 0, z: 0, heading: 0, t: 0 };
      let current = [];
      const flush = (end) => {
        if (!current.length) return;
        // The chunk's path in its own frame: starting at the origin, facing +Z.
        const c = Math.cos(from.heading);
        const sn = Math.sin(from.heading);
        const rel = current.map((p) => {
          const dx = p.x - from.x;
          const dz = p.z - from.z;
          return { x: c * dx - sn * dz, z: sn * dx + c * dz, heading: p.heading - from.heading, t: p.t - from.t };
        });
        const seconds = Math.min(KIMODO_MAX, rel[rel.length - 1].t + 0.6);
        chunks.push({
          text,
          seconds,
          path: {
            frames: [0, ...rel.map((p) => Math.round(p.t * SAMPLE_RATE))],
            xz: [[0, 0], ...rel.map((p) => [p.x, p.z])],
            heading: [0, ...rel.map((p) => p.heading)],
          },
        });
        from = end;
        current = [];
      };
      let t = 0;
      for (const p of points) {
        t += p.seconds;
        const q = { ...p, t };
        if (current.length && q.t - from.t > KIMODO_MAX - 0.8) flush(current[current.length - 1]);
        current.push(q);
      }
      flush(null);
      return chunks;
    }
    // A jump needs time to crouch, fly and land: Kimodo given 1 s only crouches (measured; 2.5 s and more jump).
    const seconds = step.action === 'stand' ? Math.min(KIMODO_MAX, Math.max(1.5, step.seconds ?? (last ? 8 : DEFAULT_STAND)))
      : step.action === 'jump' ? Math.min(KIMODO_MAX, Math.max(2.5, step.seconds ?? DEFAULT_JUMP))
        : Math.min(KIMODO_MAX, step.seconds ?? 8);
    // A dance longer than one clip, or open-ended, continues in further clips (see fromKimodo).
    return [{ text, seconds }];
  }

  /** Start generating a step's next clip (if not already). */
  request(step, index) {
    const k = step.k;
    if (!this.kimodo?.ready || k.pending || k.failed) return;
    k.chunks ??= this.kimodoChunks(step, index);
    if (k.next >= k.chunks.length) {
      // Open-ended dance (or a long one): another clip, with a new seed.
      const last = index === this.steps.length - 1;
      const total = k.chunks.reduce((t, c) => t + c.seconds, 0);
      if (!(step.action === 'dance' && (last || (step.seconds ?? 0) > total))) return;
      k.chunks.push({ ...k.chunks[k.chunks.length - 1], path: undefined });
    }
    const chunk = k.chunks[k.next++];
    const run = this.run;
    const seed = k.seed + k.next;
    this.onKimodo?.({ state: 'asking', step: index, text: chunk.text });
    k.pending = this.kimodo.generate({ ...chunk, seed }).then((r) => {
      if (run !== this.run) return;
      k.queue.push({ ...r, chunk });
      k.pending = null;
      this.onKimodo?.({ state: 'ready', step: index, text: chunk.text, ms: r.ms });
    }, (err) => {
      if (run !== this.run) return;
      k.pending = null;
      k.failed = true;
      this.onKimodo?.({ state: 'failed', step: index, text: chunk.text, error: err.message || String(err) });
    });
  }

  /** Targets from a step's Kimodo clip; a hold while it is being generated; null when Kimodo failed. */
  fromKimodo(step, last, place, pose, phrase) {
    const k = step.k;
    const index = this.steps.indexOf(step);
    if (!k.queue.length) {
      this.request(step, index);
      if (k.failed) return null;
      if (k.next >= (k.chunks?.length ?? 0) && !k.pending) {
        // Nothing more to come: the step is over, except the last one, which holds its final pose.
        if (!last || step.action !== 'stand') {
          k.finished = true;
          return this.next(pose);
        }
        return this.hold(pose, 1.8, phrase, step);
      }
      return this.hold(pose, 0.8, phrase, step, 'waiting for Kimodo');
    }
    const { clip, chunk, ms } = k.queue.shift();
    // Ask for what follows while this one plays.
    this.request(step, index);
    const transforms = [];
    if (step.action === 'walk' && k.rotation === undefined) {
      // Turn the whole pattern, as little as needed, to stay on stage (later chunks follow on from it).
      k.rotation = fitOnStage(place, step, KIMODO_GAIT).rotation;
      if (k.rotation) transforms.push(`turned ${k.rotation}° to stay on stage`);
      place = { ...place, heading: place.heading + k.rotation * DEG };
    }
    const targets = this.keyframes(clip, step.action, place);
    this.planned += clip.duration;
    if (!k.queue.length && !k.pending && k.next >= k.chunks.length && !(last && step.action === 'dance')) k.finished = !(last && step.action === 'stand');
    const entry = this.entry(`Kimodo: ${chunk.text}`, { name: 'Kimodo (NVIDIA), generated from text' },
      ['as asked'], [`${clip.duration.toFixed(1)} s generated in ${(ms / 1000).toFixed(1)} s`, `${targets.length} keyframes for MotionBricks`, ...transforms], phrase, targets.length);
    entry.movement = `kimodo:${step.action}`;
    return { targets, entry };
  }

  /**
   * MotionBricks targets through a clip, placed at the dancer: its start, then
   * keys about every `spacing` seconds, plus the highest and lowest moments of
   * the hips (a jump's top, a drop's bottom), at least 0.6 s apart.
   */
  keyframes(clip, action, place) {
    const sk = this.skeleton;
    const spacing = { stand: 1.6, walk: 1.0, jump: 0.8, dance: 0.9 }[action] ?? 1;
    const n = Math.floor(clip.duration * SAMPLE_RATE);
    const hips = [];
    const pose = new Float64Array(this.J * TF);
    for (let f = 0; f <= n; f++) hips.push(clip.sample(f / SAMPLE_RATE, pose)[1]);
    const extremes = [];
    for (let f = 3; f < n - 3; f++) {
      // A local top or bottom (within 0.2 s) that stands out by 5 cm within half a second.
      const near = hips.slice(Math.max(0, f - 6), f + 7);
      const around = hips.slice(Math.max(0, f - 15), f + 16);
      const top = hips[f] === Math.max(...near) && hips[f] - Math.min(...around) > 0.05;
      const bottom = hips[f] === Math.min(...near) && Math.max(...around) - hips[f] > 0.05;
      if (top || bottom) extremes.push(f / SAMPLE_RATE);
    }
    const end = Math.max(0, clip.duration - (POSES_PER_TOKEN - 1) / SAMPLE_RATE);
    const uniform = [];
    for (let t = spacing; t < end - 0.3; t += spacing) uniform.push(t);
    const keys = [0];
    for (const t of [...extremes, ...uniform, end].sort((a, b) => a - b)) {
      const isExtreme = extremes.includes(t);
      const prev = keys[keys.length - 1];
      if (t - prev >= 0.6) keys.push(t);
      else if (isExtreme && keys.length > 1 && t - keys[keys.length - 2] >= 0.6) keys[keys.length - 1] = t; // keep the extreme
    }
    const w0 = sk.world(clip.sample(0));
    const rl = sk.jointIndex('RightLeg');
    const ll = sk.jointIndex('LeftLeg');
    const from = { x: w0[0], z: w0[2], heading: Math.atan2(w0[rl * TF + 2] - w0[ll * TF + 2], -(w0[rl * TF] - w0[ll * TF])) };
    return keys.map((t, i) => ({
      poses: this.place(this.window({ clip }, t), from, place.x, place.z, place.heading),
      numPoses: POSES_PER_TOKEN,
      tokens: tokensFor(i === 0 ? 1.0 : t - keys[i - 1]),
      early: 0,
      movement: `kimodo:${action}`,
    }));
  }

  /** Hold the current pose (while waiting, or after the last step). */
  hold(pose, seconds, phrase, step, why = 'holding the pose') {
    if (!pose) return this.stand({ x: 0, z: 0, heading: 0 }, 1, phrase);
    const one = new Float64Array(POSES_PER_TOKEN * this.J * TF);
    for (let i = 0; i < POSES_PER_TOKEN; i++) one.set(pose, i * this.J * TF);
    let entry = null;
    if (step && step.k.announcedHold !== why) {
      step.k.announcedHold = why;
      entry = this.entry(why === 'waiting for Kimodo' ? 'Waiting for Kimodo…' : 'Holding', null, [why], [], phrase, 1);
    }
    return { targets: [{ poses: one, numPoses: POSES_PER_TOKEN, tokens: tokensFor(seconds), early: 0, movement: 'hold' }], entry };
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
      const fr = a.frames[Math.round(standing.rest * SAMPLE_RATE)];
      const from = { x: fr.root[0], z: fr.root[2], heading: fr.heading };
      // Hold one calm pose; every few holds, shift the weight to the other
      // leg and stay there. New targets from moment to moment make
      // MotionBricks hurry from one to the next, which reads as restless.
      if (!this.announced) this.holds = 0;
      windows = [];
      for (let i = 0; i < count; i++) {
        const t = Math.floor(this.holds++ / SHIFT_EVERY) % 2 ? standing.shift : standing.rest;
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

  /** One jump from learned airborne movement, placed at the dancer (chance among the best few, avoiding repeats). */
  jump(place, phrase) {
    const { jumps } = this.material();
    if (!jumps?.length) return null;
    const imp = this.improviser;
    const fresh = jumps.filter((j) => !imp.recent.slice(-3).includes(j.movement.id));
    const pool = (fresh.length ? fresh : jumps).slice(0, 4);
    const pick = pool[Math.floor(imp.random() * pool.length)];
    const m = pick.movement;
    const v = { mirrored: imp.random() < 0.35, retro: false };
    imp.recent.push(m.id);
    // Aim at the top of the jump: the movement's own key moments sit on the
    // floor (for smooth handovers), and MotionBricks only leaves the floor
    // when a target is in the air. Take-off and landing stay where the
    // movement starts and ends, if they are far enough from the top.
    const keys = [pick.peak];
    if (pick.peak - m.start >= 0.4) keys.unshift(m.start);
    if (m.end - pick.peak >= 0.4) keys.push(m.end);
    const { targets, transforms } = imp.realise({ ...m, keys }, v, place, 1);
    transforms.push('through the top of the jump');
    this.planned += m.duration;
    const entry = this.entry(m.name, m.source, ['as asked', `leaves the floor by ${pick.lift.toFixed(2)} m`], transforms, phrase, targets.length);
    entry.movement = m.id;
    return { targets, entry };
  }

  /** Walk a floor pattern from the dancer's place: one target per gait cycle along it. */
  walk(place, step, phrase) {
    const { walking } = this.material();
    if (!walking) return null;
    const { points, rotation, mirrored } = fitOnStage(place, step, walking);
    const window = this.window(walking.source, walking.time);
    const a = analysisOf(walking.source, this.skeleton);
    const fr = a.frames[Math.round(walking.time * SAMPLE_RATE)];
    const from = { x: fr.root[0], z: fr.root[2], heading: walking.heading };
    const targets = points.map((p) => ({ poses: this.place(window, from, p.x, p.z, p.heading), numPoses: POSES_PER_TOKEN, tokens: tokensFor(p.seconds), early: 0, movement: 'walk' }));
    step.done = true;
    this.planned += points.reduce((t, p) => t + p.seconds, 0);
    const transforms = [`${points.length} targets along the path`];
    if (rotation) transforms.push(`turned ${rotation}° to stay on stage`);
    if (mirrored) transforms.push('curving the other way to stay on stage');
    const entry = this.entry(`Walking ${describeStep(step).replace(/^walk: /, '')}`, walking.source, ['as asked'], transforms, phrase, targets.length);
    return { targets, entry };
  }
}
