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

import { TF, tf, tmul, getTf, setTf, tblend, qFromAxisAngle } from '../engine/math.js';
import { SAMPLE_RATE, POSES_PER_TOKEN, MIN_TOKENS, MAX_TOKENS } from '../engine/soma.js';
import { placePoses } from './transforms.js';
import { analyse } from './repertoire.js';
import { parsePrompt } from './intentions.js';
import { StageEnvironment, applySeatedLayer, applyArmReachIK, mouthPosition } from './environment.js';

export const ACTIONS = [
  'stand', 'walk', 'run', 'go_to', 'sit', 'stand_up',
  'pick_up', 'drink', 'put_down', 'wave', 'reach', 'crouch',
  'spin', 'jump', 'dance', 'imitate',
];
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
const STAND_UP = /\b(stand(?:s|ing)? up|get(?:s|ting)? up|rise(?:s|ing)?(?: up)?|leave (?:the )?(?:chair|seat))\b/;
const SIT = /\b(sit(?:s|ting)?(?: down| on (?:the )?(?:chair|seat|stool))?|take a seat|seat(?:ed)?)\b/;
const PICK_UP = /\b(pick(?:s|ing)? up|grab(?:s|bing)?|take(?:s|ing)?|get(?:s|ting)?|lift(?:s|ing)?)\b(?:\s+(?:the|a|an|my))?\s*\b(glass|mug|cup|drink|water|soda|coffee|bottle|object|item|floor)?\b/;
const DRINK = /\b(drink(?:s|ing)?|sip(?:s|ping)?|take a (?:drink|sip)|quaff)\b/;
const PUT_DOWN = /\b(put(?:s|ting)? (?:it )?down|set(?:s|ting)? (?:it |the (?:glass|mug|cup) )?down|place(?:s|ing)? (?:it |the (?:glass|mug|cup) )?(?:down|back|on (?:the )?table)|return (?:the )?(?:glass|mug|cup))\b/;
const GO_TO = /\b(go(?:es|ing)? to|walk(?:s|ing)? (?:over )?to|move(?:s|ing)? to|head(?:s|ing)? (?:over )?to|approach(?:es|ing)?)\s+(?:the\s+)?(table|chair|seat|glass|mug|cup|center|centre|stage left|stage right|left|right|front|back|camera|downstage|upstage)\b/;
const RUN = /\b(run(?:s|ning)?|sprint(?:s|ing)?|dash(?:es|ing)?|jog(?:s|ging)?)\b/;
const WALK = /\b(walk(?:s|ing)?|stroll(?:s|ing)?|pac(?:e|ing)|go(?:es)? (?:round|around)|step(?:s|ping)? (?:forward|around))\b/;
const CIRCLE = /\b(circles?|circular|round|around|loop)\b/;
const TURN = /^(?:turn|rotate|face)\b/;
const SPIN = /^(?:do (?:a |some )?|make (?:a |some )?)?(spin|pirouette|twirl|whirl)(?:s|ing)?\b/;
// A jump as an action: the clause starts with it ("jump", "then jump twice"), not "ballet with jumps".
const JUMP = /^(?:do (?:a |some )?|make (?:a |some )?)?(jump|hop|leap|jete|jeté|grand jete)(?:s|ing)?\b/;
const WAVE = /^(?:do (?:a )?)?(wave|greet|gesture|signal|beckon|bow)(?:s|ing)?\b/;
const REACH = /^(?:reach(?:es|ing)?(?:\s+(?:up|high|out|for the sky))?|tiptoe)\b/;
const CROUCH = /^(?:crouch(?:es|ing)?|kneel(?:s|ing)?|squat(?:s|ting)?|duck(?:s|ing)?|pick up from (?:the )?floor)\b/;
const IMITATE = /\b(copy(?:ing)? (?:my |the |your )?(?:moves?|movements?|motion|pose|dance)|mirror(?:ing)? me|imitate(?:s|ing)?|follow(?:ing)? my moves?|improvise based on (?:them|my moves?|it)|riff on (?:my moves?|them))\b/;
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
  if (IMITATE.test(clause)) {
    const mode = /\b(improvise|riff|vary|develop)\b/.test(clause) && !/\bcopy|mirror|imitate\b/.test(clause)
      ? 'riff'
      : /\b(improvise|riff|vary|develop)\b/.test(clause) ? 'copy_and_riff' : 'copy';
    return step('imitate', { mode, seconds: readSeconds(clause) });
  }
  const goMatch = clause.match(GO_TO);
  if (goMatch) {
    const isRun = RUN.test(clause);
    return step(isRun ? 'run' : 'go_to', { target: goMatch[2] });
  }
  if (SIT.test(clause)) return step('sit', { target: 'chair', seconds: readSeconds(clause) });
  if (STAND_UP.test(clause)) return step('stand_up', { seconds: readSeconds(clause) });
  if (PUT_DOWN.test(clause)) return step('put_down', { target: 'table', seconds: readSeconds(clause) });
  if (DRINK.test(clause) && !/\bpick up|grab|take\b/.test(clause)) {
    return step('drink', { target: 'glass', seconds: readSeconds(clause) });
  }
  if (PICK_UP.test(clause) && /^(?:please\s+)?(?:pick|grab|take|get|lift)\b/.test(clause)) {
    const m = clause.match(PICK_UP);
    const target = /\b(floor|ground)\b/.test(clause) || m?.[2] === 'floor' ? 'floor' : 'glass';
    return step('pick_up', { target, seconds: readSeconds(clause) });
  }
  if (RUN.test(clause) && /^(?:please\s+)?(?:run|sprint|dash|jog)\b/.test(clause)) {
    if (CIRCLE.test(clause)) {
      let radius = readMetres(clause);
      if (radius === null) radius = /\b(small|tight|little)\b/.test(clause) ? 1 : /\b(big|large|wide)\b/.test(clause) ? 2.5 : DEFAULT_RADIUS;
      const side = readSide(clause);
      return step('run', { moves: [{ move: 'arc', radius, angle: 360 * (side ?? 1), distance: null, free: side === null }], repeat: readTimes(clause) });
    }
    return step('run', { moves: [{ move: 'forward', distance: readMetres(clause) ?? DEFAULT_DISTANCE, angle: null, radius: null }] });
  }
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
  if (SPIN.test(clause)) return step('spin', { seconds: readSeconds(clause) });
  if (TURN.test(clause)) {
    const m = clause.match(/(\d+(?:\.\d+)?)\s*(?:°|deg(?:rees?)?)/);
    const angle = m ? Number(m[1]) : /\b(around|about)\b/.test(clause) ? 180 : 90;
    return step('walk', { moves: [{ move: 'turn', angle: angle * (readSide(clause) ?? 1), distance: null, radius: null }] });
  }
  if (JUMP.test(clause)) {
    const times = readTimes(clause);
    return step('jump', { seconds: readSeconds(clause) ?? (times ? times * 1.5 : null) });
  }
  if (WAVE.test(clause)) return step('wave', { seconds: readSeconds(clause) });
  if (REACH.test(clause)) return step('reach', { seconds: readSeconds(clause) });
  if (CROUCH.test(clause)) return step('crouch', { seconds: readSeconds(clause) });
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
  const rawClauses = text.split(SPLIT).map((c) => c.replace(/^\s*(first|to start|start by|begin by)\b/, '').trim()).filter(Boolean);
  const clauses = [];
  for (const clause of rawClauses) {
    // Split comma/and-separated action lists ("go to the table, sit, pick up the glass and drink")
    // only when the first item is itself a directed action (so quality prompts like
    // "slow and low, like sinking into water" or "vigorous ballet with jumps and turns" stay intact).
    const parts = clause.split(/,\s*(?:and\s+)?|\s+\band\s+/).map((p) => p.trim()).filter(Boolean);
    if (parts.length > 1 && readStep(parts[0]) !== null && parts.every((p) => readStep(p) !== null || /^(?:dance|improvise)\b/.test(p))) {
      clauses.push(...parts);
    } else {
      clauses.push(clause);
    }
  }
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
  return raw.slice(0, 16).filter((s) => ACTIONS.includes(s?.action)).map((s) => {
    const out = { action: s.action, seconds: n(s.seconds, 0.5, 600), moves: null, repeat: null };
    if (typeof s.text === 'string' && s.text.trim()) out.text = s.text.trim().slice(0, 300);
    if (typeof s.target === 'string' && s.target.trim()) out.target = s.target.trim().slice(0, 60);
    else if (s.target && typeof s.target === 'object' && Number.isFinite(s.target.x) && Number.isFinite(s.target.z)) {
      out.target = {
        x: n(s.target.x, -STAGE_RADIUS - 1, STAGE_RADIUS + 1),
        z: n(s.target.z, -STAGE_RADIUS - 1, STAGE_RADIUS + 1),
        ...(Number.isFinite(s.target.heading) ? { heading: s.target.heading } : {}),
        ...(typeof s.target.label === 'string' ? { label: s.target.label.slice(0, 60) } : {}),
      };
    }
    if (typeof s.mode === 'string' && s.mode.trim()) out.mode = s.mode.trim().slice(0, 40);
    if (s.action === 'walk' || (s.action === 'run' && !out.target)) {
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
  const targetLabel = typeof s.target === 'string' ? s.target : s.target?.label || (s.target ? `(${fmt(s.target.x)}, ${fmt(s.target.z)})` : '');
  if (s.action === 'stand') return `stand${secs}`;
  if (s.action === 'stand_up') return `stand up${secs}`;
  if (s.action === 'sit') return `sit${targetLabel ? ` on ${targetLabel}` : ''}${secs}`;
  if (s.action === 'go_to') return `go to ${targetLabel || 'table'}`;
  if (s.action === 'pick_up') return `pick up ${targetLabel || 'glass'}`;
  if (s.action === 'drink') return `drink${secs}`;
  if (s.action === 'put_down') return `put down glass`;
  if (s.action === 'wave') return `wave${secs}`;
  if (s.action === 'reach') return `reach high${secs}`;
  if (s.action === 'crouch') return `crouch${secs}`;
  if (s.action === 'spin') return `spin${secs}`;
  if (s.action === 'jump') return `jump${secs}`;
  if (s.action === 'imitate') return s.mode === 'riff' ? `improvise on partner${secs}` : s.mode === 'copy_and_riff' ? `copy & improvise${secs}` : `copy moves${secs}`;
  if (s.action === 'walk' || s.action === 'run') {
    if (targetLabel && !s.moves?.length) return `${s.action} to ${targetLabel}`;
    const side = (a) => (a >= 0 ? 'left' : 'right');
    const parts = (s.moves || []).map((m) => (m.move === 'forward' ? `forward ${fmt(m.distance)} m`
      : m.move === 'turn' ? `turn ${side(m.angle)} ${fmt(Math.abs(m.angle))}°`
        : `arc ${m.free ? '' : `${side(m.angle)} `}${fmt(Math.abs(m.angle))}° r ${fmt(m.radius)} m`));
    return `${s.action}: ${parts.join(', ')}${s.repeat > 1 ? `, ×${s.repeat}` : ''}`;
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

const ENV_ACTIONS = new Set(['go_to', 'sit', 'stand_up', 'pick_up', 'drink', 'put_down', 'imitate', 'wave', 'reach', 'crouch', 'spin', 'run']);
const ONE_SHOT_ACTIONS = new Set(['walk', 'run', 'go_to', 'sit', 'stand_up', 'pick_up', 'drink', 'put_down', 'wave', 'reach', 'crouch', 'spin']);

export class Director {
  /**
   * @param {{ improviser: import('./improviser.js').Improviser, skeleton: object, kimodo?: import('./kimodo.js').KimodoClient, env?: StageEnvironment }} opts
   */
  constructor({ improviser, skeleton, kimodo = null, env = null }) {
    this.improviser = improviser;
    this.kimodo = kimodo; // text-to-motion for every step, when its server is running
    this.env = env || new StageEnvironment();
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
    this.running = null;
    this.materialFrom = 0;
    this.imitateStep = 0;
    this.lastCopied = null;
  }

  /**
   * A new score (steps may be []: pure improvisation with the intentions).
   * With Kimodo, general text steps are generated from text: a prompt without steps
   * becomes one open-ended dance, and a score ending in a walk or a jump
   * ends standing. Environment-aware interactions run directly against this.env.
   */
  setScore(steps, intentions, { prompt = '', summary = '' } = {}) {
    this.steps = (steps || []).map((s) => ({ ...s }));
    this.intentions = intentions;
    this.prompt = prompt;
    this.index = 0;
    this.planned = 0;
    this.announced = false;
    this.imitateStep = 0;
    this.run = (this.run || 0) + 1;
    if (this.kimodo?.ready) {
      if (!this.steps.length) this.steps.push({ action: 'dance', seconds: null, moves: null, repeat: null, text: null, free: true, summary });
      if (['walk', 'jump'].includes(this.steps[this.steps.length - 1].action) && !this.steps[this.steps.length - 1].target) {
        this.steps.push({ action: 'stand', seconds: null, moves: null, repeat: null, text: null, implied: true });
      }
      this.steps.forEach((step, i) => {
        if (ENV_ACTIONS.has(step.action) || (step.action === 'walk' && step.target)) return;
        step.k = { queue: [], pending: null, failed: false, finished: false, chunks: null, next: 0, seed: (this.improviser.seed * 7919 + i * 101) % 100000 };
      });
      // Ask for every step now (the server takes them in order), so later steps are ready in time.
      this.steps.forEach((step, i) => step.k && this.request(step, i));
    }
    this.improviser.setIntentions(this.intentionsFor(this.steps[0]));
  }

  /** What the improviser works with during a step. */
  intentionsFor(step) {
    const base = step?.intentions ?? this.intentions ?? {};
    if (step?.action === 'jump') return { ...base, jump: 1, energy: Math.max(base.energy ?? 0, 0.75), stillness: 0, travel: null, variety: 0.3, tempo: 1 };
    if (step?.action === 'spin') return { ...base, turn: 1, energy: Math.max(base.energy ?? 0, 0.7), tempo: 1 };
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
        : ONE_SHOT_ACTIONS.has(s.action) ? (s.done ? 0 : Infinity)
        : s.action === 'jump' ? (s.seconds ?? DEFAULT_JUMP)
        : s.action === 'imitate' ? (s.seconds ?? (last ? Infinity : 6))
          : last ? Infinity : (s.seconds ?? (s.action === 'stand' ? DEFAULT_STAND : DEFAULT_DANCE));
      if (this.planned < limit) return { step: s, last };
      this.index++;
      this.planned = 0;
      this.announced = false;
      const next = this.steps[this.index];
      if (next) this.improviser.setIntentions(this.intentionsFor(next));
    }
    // After a directed score: stay seated if on the chair, or stand where it ended.
    return { step: { action: 'stand', seconds: null, implied: true }, last: true };
  }

  /**
   * The next targets, the same contract as Improviser.next.
   * @param {Float64Array|null} pose  the last generated pose (null at the start)
   */
  next(pose) {
    if (!this.steps.length) {
      if (this.env.seated && pose) {
        // Stand up before free dancing if previously seated.
        return this.standUp(this.improviser.sense(pose).place, { action: 'stand_up' }, 'stand up', false);
      }
      return this.improviser.next(pose);
    }
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
    } else if (step.action === 'go_to') {
      const out = this.goTo(place, step, phrase, false);
      if (out) return out;
    } else if (step.action === 'run') {
      const out = step.target ? this.goTo(place, step, phrase, true) : this.walk(place, step, phrase, true);
      if (out) return out;
    } else if (step.action === 'sit') {
      const out = this.sit(place, step, phrase, pose);
      if (out) return out;
    } else if (step.action === 'stand_up') {
      const out = this.standUp(place, step, phrase, true);
      if (out) return out;
    } else if (step.action === 'pick_up') {
      const out = this.pickUp(place, step, phrase, pose);
      if (out) return out;
    } else if (step.action === 'drink') {
      const out = this.drink(place, step, phrase, pose);
      if (out) return out;
    } else if (step.action === 'put_down') {
      const out = this.putDown(place, step, phrase, pose);
      if (out) return out;
    } else if (step.action === 'spin') {
      const out = this.spin(place, step, phrase);
      if (out) return out;
    } else if (step.action === 'wave' || step.action === 'reach' || step.action === 'crouch') {
      const out = this.expressiveAction(place, step, phrase, pose);
      if (out) return out;
    } else if (step.action === 'imitate') {
      const out = this.imitate(place, step, phrase, pose);
      if (out) return out;
    } else if (step.action === 'jump') {
      if (this.env.seated) return this.standUp(place, step, phrase, false);
      const out = this.jump(place, phrase);
      if (out) return out;
    } else if (step.action === 'walk') {
      if (step.target) {
        const out = this.goTo(place, step, phrase, false);
        if (out) return out;
      }
      if (this.env.seated) return this.standUp(place, step, phrase, false);
      const out = this.walk(place, step, phrase, false);
      if (out) return out;
      // No walking material: travel with the improviser instead.
      step.done = true;
      this.improviser.setIntentions({ ...this.intentions, travel: 0.95, jump: 0, turn: 0.2 });
    }
    if (this.env.seated) return this.standUp(place, step, phrase, false);
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

  /** A single calm local pose placed at `place`, using learned standing material or the current/reference pose. */
  basePose(place, pose = null) {
    const { standing } = this.material();
    if (standing) {
      const source = standing.source;
      const a = analysisOf(source, this.skeleton);
      const fr = a.frames[Math.round(standing.rest * SAMPLE_RATE)];
      const from = { x: fr.root[0], z: fr.root[2], heading: fr.heading };
      const one = source.clip.sample(standing.rest);
      return this.place(one, from, place.x, place.z, place.heading);
    }
    if (pose) return Float64Array.from(pose);
    const ref = Float64Array.from(this.skeleton.reference);
    ref[0] = place.x;
    ref[2] = place.z;
    return ref;
  }

  /** Build a 30 Hz directPoses target with a 4-pose endpoint window and synchronous + post-alignment IK modifier. */
  makeDirectTarget(frameCount, movement, buildFrame) {
    const J = this.J;
    const directPoses = new Float64Array(frameCount * J * TF);
    const directContacts = new Float32Array(frameCount * 4);
    const propFrames = [];
    const blendFrames = Math.min(10, Math.floor(frameCount / 3));
    const runBuild = (posesArr, contactsArr, preserveFirst = false) => {
      const cache = {};
      propFrames.length = 0;
      const firstPose = preserveFirst ? Float64Array.from(posesArr.subarray(0, J * TF)) : null;
      for (let f = 0; f < frameCount; f++) {
        const local = posesArr.subarray(f * J * TF, (f + 1) * J * TF);
        const c = contactsArr.subarray(f * 4, f * 4 + 4);
        c.set([1, 1, 1, 1]);
        buildFrame(local, f, frameCount, c, cache);
        if (preserveFirst && firstPose && f < blendFrames) {
          const u = f / blendFrames;
          const alpha = u * u * (3 - 2 * u);
          for (let j = 0; j < J; j++) {
            setTf(local, j, tblend(getTf(firstPose, j), getTf(local, j), alpha));
          }
        }
        this.env.updatePropsFromPose(this.skeleton, local);
        propFrames.push({
          heldBy: this.env.glass.heldBy,
          holding: this.env.holding,
          seated: this.env.seated,
          x: this.env.glass.x,
          y: this.env.glass.y,
          z: this.env.glass.z,
          tilt: this.env.glass.tilt,
        });
      }
    };
    runBuild(directPoses, directContacts, false);
    const poses = new Float64Array(POSES_PER_TOKEN * J * TF);
    for (let i = 0; i < POSES_PER_TOKEN; i++) {
      const f = Math.max(0, frameCount - POSES_PER_TOKEN + i);
      poses.set(directPoses.subarray(f * J * TF, (f + 1) * J * TF), i * J * TF);
    }
    const seconds = frameCount / SAMPLE_RATE;
    return {
      poses,
      directPoses,
      directContacts,
      propFrames,
      modifier: (outPoses, outContacts) => runBuild(outPoses, outContacts, true),
      numPoses: POSES_PER_TOKEN,
      tokens: tokensFor(seconds),
      early: 0,
      movement,
    };
  }

  /** Hold a natural standing (or seated) pose at the dancer's place: `count` targets. */
  stand(place, count, phrase = 'start', pose = null) {
    if (this.env.seated) {
      const chair = this.env.chair;
      const anchor = chair.seatAnchor;
      const holdPos = [1.25, 0.86, -1.14];
      const holdingGlass = this.env.holding === 'glass';
      const rightLap = [chair.seatAnchor.x - 0.16, chair.seatAnchor.y + 0.08, chair.seatAnchor.z + 0.25];
      const targets = Array.from({ length: count }, () => this.makeDirectTarget(Math.round(HOLD * SAMPLE_RATE), 'stand', (local, f, n, _c, cache) => {
        local.set(this.basePose(anchor, pose));
        applySeatedLayer(this.skeleton, local, chair, 1, { leftArmRest: true, rightArmRest: !holdingGlass, cache });
        const breath = 0.04 + 0.004 * Math.sin((f / Math.max(1, n - 1)) * 2 * Math.PI);
        if (holdingGlass) {
          applyArmReachIK(this.skeleton, local, holdPos, 1, { hand: 'right', spineLean: breath, headPitch: 0, cache });
        } else {
          applyArmReachIK(this.skeleton, local, rightLap, 0.85, { hand: 'right', spineLean: breath, headPitch: 0, cache });
        }
      }));
      this.planned += count * HOLD;
      let entry = null;
      if (!this.announced) {
        this.announced = true;
        entry = this.entry('Seated at the table', null, ['as asked'], [this.env.holding === 'glass' ? 'seated, holding the glass' : 'seated on the chair'], phrase, targets.length);
      }
      return { targets, entry };
    }
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

  /** Navigate to a named stage prop (table, chair, glass, center) or clicked coordinate {x, z}. */
  goTo(place, step, phrase, isRun = false) {
    const dest = this.env.resolveTarget(step.target, place);
    this.env.beacon = { active: true, x: dest.x, z: dest.z, label: dest.label };
    if (this.env.seated) {
      // If asked to go somewhere away from the chair/table while seated, stand up first.
      if (Math.hypot(dest.x - this.env.chair.approachAnchor.x, dest.z - this.env.chair.approachAnchor.z) > 0.35) {
        return this.standUp(place, { action: 'stand_up' }, phrase, false);
      }
      step.done = true;
      return this.stand(place, 1, phrase);
    }
    const moves = this.env.planMovesTo(place, dest);
    if (!moves.length) {
      step.done = true;
      this.anchor = { x: dest.x, z: dest.z, heading: dest.heading };
      return this.stand(this.anchor, 1, phrase);
    }
    const walkStep = { ...step, action: isRun ? 'run' : 'walk', moves, repeat: 1, noFitStage: true, destLabel: dest.label };
    const out = this.walk(place, walkStep, phrase, isRun);
    if (out) {
      step.done = true;
      return out;
    }
    // Fallback when no walking clip is loaded yet: generate a direct locomotion transition to dest.
    const dist = Math.hypot(dest.x - place.x, dest.z - place.z);
    const secs = Math.max(0.8, dist / (isRun ? 2.2 : 1.15));
    const frames = Math.max(24, Math.round(secs * SAMPLE_RATE));
    const target = this.makeDirectTarget(frames, isRun ? 'run' : 'walk', (local, f, n) => {
      const u = n > 1 ? f / (n - 1) : 1;
      const s = u * u * (3 - 2 * u);
      const h = place.heading + wrapAngle(dest.heading - place.heading) * s;
      const p = { x: place.x + (dest.x - place.x) * s, z: place.z + (dest.z - place.z) * s, heading: h };
      local.set(this.basePose(p, null, f / SAMPLE_RATE));
    });
    step.done = true;
    this.planned += secs;
    const entry = this.entry(`${isRun ? 'Running' : 'Walking'} to ${dest.label}`, null, ['as asked'], [`arrived at ${dest.label}`], phrase, 1);
    return { targets: [target], entry };
  }

  /** Sit down on the chair (walking to the chair first if not already there). */
  sit(place, step, phrase, pose = null) {
    const chair = this.env.chair;
    const approach = chair.approachAnchor;
    if (!this.env.seated && Math.hypot(place.x - approach.x, place.z - approach.z) > 0.45) {
      // Approach the chair first; step.done remains false so sit runs upon arrival.
      return this.goTo(place, { action: 'go_to', target: 'chair' }, phrase, false);
    }
    const frames = 45; // 1.5 s smooth sit-down
    const wasSeated = this.env.seated;
    const holdingGlass = this.env.holding === 'glass';
    const holdPos = [1.25, 0.86, -1.14];
    const target = this.makeDirectTarget(frames, 'sit', (local, f, n, c, cache) => {
      const u = wasSeated ? 1 : n > 1 ? f / (n - 1) : 1;
      if (!wasSeated && u < 0.94) c.set([0, 0, 0, 0]);
      const curPlace = {
        x: place.x + (approach.x - place.x) * Math.min(1, u * 2),
        z: place.z + (approach.z - place.z) * Math.min(1, u * 2),
        heading: place.heading + wrapAngle(chair.heading - place.heading) * Math.min(1, u * 2),
      };
      local.set(this.basePose(curPlace, pose));
      this.env.seated = u >= 0.85;
      applySeatedLayer(this.skeleton, local, chair, u, { leftArmRest: true, rightArmRest: !holdingGlass, cache });
      // Natural forward torso counterbalance as the hips lower onto the chair seat, settling to 0.04.
      const s = u * u * (3 - 2 * u);
      const balanceLean = wasSeated ? 0.04 : Math.sin(u * Math.PI) * 0.24 + 0.04 * s;
      const rightLap = [chair.seatAnchor.x - 0.16, chair.seatAnchor.y + 0.08, chair.seatAnchor.z + 0.25];
      if (holdingGlass) {
        applyArmReachIK(this.skeleton, local, holdPos, 1, { hand: 'right', spineLean: balanceLean, headPitch: 0, cache });
      } else {
        applyArmReachIK(this.skeleton, local, rightLap, s * 0.85, { hand: 'right', spineLean: balanceLean, headPitch: 0, cache });
      }
    });
    this.env.seated = true;
    this.anchor = { x: chair.seatAnchor.x, z: chair.seatAnchor.z, heading: chair.heading };
    step.done = true;
    this.planned += frames / SAMPLE_RATE;
    const entry = this.entry('Sitting on the chair', this.standing?.source || null, ['as asked'],
      ['pelvis lowered onto chair seat (0.53 m)', 'feet planted, knees at 104°'], phrase, 1);
    return { targets: [target], entry };
  }

  /** Rise smoothly from the chair to standing. */
  standUp(place, step, phrase, markDone = true) {
    const chair = this.env.chair;
    const approach = chair.approachAnchor;
    const frames = 42; // 1.4 s smooth stand-up
    const holdingGlass = this.env.holding === 'glass';
    const target = this.makeDirectTarget(frames, 'stand_up', (local, f, n, c, cache) => {
      const u = n > 1 ? 1 - f / (n - 1) : 0; // 1 (seated) -> 0 (standing)
      if (u > 0.08) c.set([0, 0, 0, 0]);
      local.set(this.basePose(approach, null, f / SAMPLE_RATE));
      this.env.seated = u > 0.2;
      applySeatedLayer(this.skeleton, local, chair, u, { leftArmRest: true, cache });
      const balanceLean = Math.sin(u * Math.PI) * 0.30;
      if (holdingGlass) {
        const holdPos = [1.25, 1.12 - 0.26 * u, -1.14];
        applyArmReachIK(this.skeleton, local, holdPos, 0.85, { hand: 'right', spineLean: balanceLean, cache });
      } else if (balanceLean > 0.01) {
        const rightSide = [approach.x - 0.22, 0.90, approach.z + 0.08];
        applyArmReachIK(this.skeleton, local, rightSide, (1 - u) * 0.5, { hand: 'right', spineLean: balanceLean, cache });
      }
    });
    this.env.seated = false;
    this.anchor = { ...approach };
    if (markDone) step.done = true;
    this.planned += frames / SAMPLE_RATE;
    const entry = this.entry('Standing up from the chair', this.standing?.source || null, ['as asked'], ['weight shifted forward over feet, rose to standing'], phrase, 1);
    return { targets: [target], entry };
  }

  /** Pick up the glass from the table (or an object from the floor), using upper-body layering if seated. */
  pickUp(place, step, phrase, pose = null) {
    if (step.target === 'floor') {
      if (this.env.seated) return this.standUp(place, step, phrase, false);
      const frames = 54;
      const floorSpot = [place.x + Math.sin(place.heading) * 0.32, 0.08, place.z + Math.cos(place.heading) * 0.32];
      const target = this.makeDirectTarget(frames, 'pick_up', (local, f, n, c, cache) => {
        const u = n > 1 ? f / (n - 1) : 1;
        const dip = Math.sin(u * Math.PI);
        local.set(this.basePose(place, pose, f / SAMPLE_RATE));
        const crouchChair = {
          seatAnchor: { x: place.x, y: 1.01 - 0.42 * dip, z: place.z, heading: place.heading },
          feet: {
            left: [place.x + Math.cos(place.heading) * 0.12, 0.08, place.z - Math.sin(place.heading) * 0.12],
            right: [place.x - Math.cos(place.heading) * 0.12, 0.08, place.z + Math.sin(place.heading) * 0.12],
          },
        };
        applySeatedLayer(this.skeleton, local, crouchChair, dip, { leftArmRest: false, cache });
        applyArmReachIK(this.skeleton, local, floorSpot, dip, { hand: 'right', spineLean: 0.35 * dip, headPitch: -0.25 * dip, cache });
      });
      step.done = true;
      this.planned += frames / SAMPLE_RATE;
      const entry = this.entry('Picking up from the floor', null, ['as asked'], ['crouched and reached to the floor'], phrase, 1);
      return { targets: [target], entry };
    }

    // Picking up the glass from the table.
    const tableAnchor = this.env.table.standAnchor;
    if (!this.env.seated && Math.hypot(place.x - tableAnchor.x, place.z - tableAnchor.z) > 0.65) {
      return this.goTo(place, { action: 'go_to', target: 'table' }, phrase, false);
    }
    const chair = this.env.chair;
    const isSeated = this.env.seated;
    const glassPos = [...this.env.glass.home];
    const holdPos = [1.25, isSeated ? 0.86 : 1.12, -1.14];
    const frames = 48; // 1.6 s reach, grasp, and lift
    const target = this.makeDirectTarget(frames, 'pick_up', (local, f, n, c, cache) => {
      const u = n > 1 ? f / (n - 1) : 1;
      const baseP = isSeated ? chair.seatAnchor : tableAnchor;
      local.set(this.basePose(baseP, pose, f / SAMPLE_RATE));
      if (isSeated) applySeatedLayer(this.skeleton, local, chair, 1, { leftArmRest: true, cache });

      if (u <= 0.62) {
        // Reach toward the glass on the table.
        const w = Math.sin((u / 0.62) * (Math.PI / 2));
        this.env.glass.heldBy = null;
        this.env.holding = null;
        [this.env.glass.x, this.env.glass.y, this.env.glass.z] = glassPos;
        this.env.glass.tilt = 0;
        applyArmReachIK(this.skeleton, local, glassPos, w, {
          hand: 'right',
          spineLean: isSeated ? 0.04 + 0.20 * w : 0.18 * w,
          headPitch: -0.18 * w,
          cache,
        });
      } else {
        // Grasp the glass and lift it to a poised hold position.
        const v = (u - 0.62) / 0.38;
        const s = v * v * (3 - 2 * v);
        this.env.glass.heldBy = 'RightHand';
        this.env.holding = 'glass';
        this.env.glass.tilt = 0;
        const handGoal = [
          glassPos[0] + (holdPos[0] - glassPos[0]) * s,
          glassPos[1] + (holdPos[1] - glassPos[1]) * s,
          glassPos[2] + (holdPos[2] - glassPos[2]) * s,
        ];
        const lean = isSeated ? 0.04 + 0.20 * (1 - s) : 0.18 * (1 - s * 0.75);
        applyArmReachIK(this.skeleton, local, handGoal, 1, {
          hand: 'right',
          spineLean: lean,
          headPitch: -0.18 * (1 - s),
          cache,
        });
      }
    });
    this.env.glass.heldBy = 'RightHand';
    this.env.holding = 'glass';
    step.done = true;
    this.planned += frames / SAMPLE_RATE;
    const entry = this.entry('Picking up the glass', null, ['as asked'],
      [isSeated ? 'upper-body reach while seated on the chair' : 'reached to the table', 'right-hand IK grasped glass at (1.24, 0.76, -0.98)'], phrase, 1);
    return { targets: [target], entry };
  }

  /** Drink from the glass (picking it up from the table first if not already holding it). */
  drink(place, step, phrase, pose = null) {
    const tableAnchor = this.env.table.standAnchor;
    if (!this.env.seated && Math.hypot(place.x - tableAnchor.x, place.z - tableAnchor.z) > 0.65) {
      return this.goTo(place, { action: 'go_to', target: 'table' }, phrase, false);
    }
    if (this.env.holding !== 'glass') {
      return this.pickUp(place, { action: 'pick_up', target: 'glass' }, phrase, pose);
    }
    const chair = this.env.chair;
    const isSeated = this.env.seated;
    const holdPos = [1.25, isSeated ? 0.86 : 1.12, -1.14];
    const frames = 60; // 2.0 s bring glass to mouth, tilt to sip, and lower
    const target = this.makeDirectTarget(frames, 'drink', (local, f, n, c, cache) => {
      const u = n > 1 ? f / (n - 1) : 1;
      const baseP = isSeated ? chair.seatAnchor : tableAnchor;
      local.set(this.basePose(baseP, pose, f / SAMPLE_RATE));
      if (isSeated) applySeatedLayer(this.skeleton, local, chair, 1, { leftArmRest: true, cache });

      this.env.glass.heldBy = 'RightHand';
      this.env.holding = 'glass';
      const mouth = mouthPosition(this.skeleton, local);
      let s = 0;
      let headPitch = 0;
      let tilt = 0;
      if (u < 0.30) {
        const v = u / 0.30;
        s = v * v * (3 - 2 * v);
        tilt = 0.25 * s;
        headPitch = 0.08 * s;
      } else if (u <= 0.72) {
        s = 1;
        const sip = Math.sin(((u - 0.30) / 0.42) * Math.PI);
        tilt = 0.25 + 0.65 * sip;
        headPitch = 0.08 + 0.22 * sip;
      } else {
        const v = 1 - (u - 0.72) / 0.28;
        s = v * v * (3 - 2 * v);
        tilt = 0.20 * s;
        headPitch = 0.08 * s;
      }
      this.env.glass.tilt = tilt;
      const handGoal = [
        holdPos[0] + (mouth[0] - holdPos[0]) * s,
        holdPos[1] + (mouth[1] - holdPos[1]) * s,
        holdPos[2] + (mouth[2] - holdPos[2]) * s,
      ];
      applyArmReachIK(this.skeleton, local, handGoal, 1, {
        hand: 'right',
        spineLean: isSeated ? 0.04 * (1 - s) : 0,
        headPitch,
        cache,
      });
    });
    this.env.glass.sips++;
    this.env.glass.tilt = 0;
    step.done = true;
    this.planned += frames / SAMPLE_RATE;
    const entry = this.entry('Drinking from the glass', null, ['as asked'],
      [isSeated ? 'seated at the table' : 'standing at the table', 'brought glass to mouth and tilted to drink'], phrase, 1);
    return { targets: [target], entry };
  }

  /** Place the held glass back onto the table. */
  putDown(place, step, phrase, pose = null) {
    const tableAnchor = this.env.table.standAnchor;
    if (!this.env.seated && Math.hypot(place.x - tableAnchor.x, place.z - tableAnchor.z) > 0.65) {
      return this.goTo(place, { action: 'go_to', target: 'table' }, phrase, false);
    }
    const chair = this.env.chair;
    const isSeated = this.env.seated;
    const glassPos = [...this.env.glass.home];
    const holdPos = [1.25, isSeated ? 0.86 : 1.12, -1.14];
    const frames = 45;
    const target = this.makeDirectTarget(frames, 'put_down', (local, f, n, c, cache) => {
      const u = n > 1 ? f / (n - 1) : 1;
      const baseP = isSeated ? chair.seatAnchor : tableAnchor;
      local.set(this.basePose(baseP, pose, f / SAMPLE_RATE));
      if (isSeated) applySeatedLayer(this.skeleton, local, chair, 1, { leftArmRest: true, cache });
      if (u <= 0.58) {
        const v = u / 0.58;
        const s = v * v * (3 - 2 * v);
        this.env.glass.heldBy = 'RightHand';
        this.env.holding = 'glass';
        const handGoal = [
          holdPos[0] + (glassPos[0] - holdPos[0]) * s,
          holdPos[1] + (glassPos[1] - holdPos[1]) * s,
          holdPos[2] + (glassPos[2] - holdPos[2]) * s,
        ];
        applyArmReachIK(this.skeleton, local, handGoal, 1, { hand: 'right', spineLean: 0.04 + 0.18 * s, headPitch: -0.18 * s, cache });
      } else {
        const v = 1 - (u - 0.58) / 0.42;
        const s = v * v * (3 - 2 * v);
        this.env.glass.heldBy = null;
        this.env.holding = null;
        [this.env.glass.x, this.env.glass.y, this.env.glass.z] = glassPos;
        this.env.glass.tilt = 0;
        const rightLap = [chair.seatAnchor.x - 0.16, chair.seatAnchor.y + 0.08, chair.seatAnchor.z + 0.25];
        const goal = isSeated
          ? [
              rightLap[0] + (glassPos[0] - rightLap[0]) * s,
              rightLap[1] + (glassPos[1] - rightLap[1]) * s,
              rightLap[2] + (glassPos[2] - rightLap[2]) * s,
            ]
          : glassPos;
        applyArmReachIK(this.skeleton, local, goal, isSeated ? 0.85 + 0.15 * s : s, {
          hand: 'right',
          spineLean: (isSeated ? 0.04 : 0) + 0.18 * s,
          headPitch: -0.18 * s,
          cache,
        });
      }
    });
    this.env.glass.heldBy = null;
    this.env.holding = null;
    [this.env.glass.x, this.env.glass.y, this.env.glass.z] = glassPos;
    this.env.glass.tilt = 0;
    step.done = true;
    this.planned += frames / SAMPLE_RATE;
    const entry = this.entry('Setting the glass on the table', null, ['as asked'], ['placed glass back on table'], phrase, 1);
    return { targets: [target], entry };
  }

  /** Spin / pirouette from the repertoire (or a 360-degree turn in place). */
  spin(place, step, phrase) {
    if (this.env.seated) return this.standUp(place, step, phrase, false);
    const imp = this.improviser;
    const turns = [...imp.movements].sort((a, b) => Math.abs(b.turn) - Math.abs(a.turn));
    if (turns.length && Math.abs(turns[0].turn) > 1.2) {
      const pick = turns[0];
      const { targets, transforms } = imp.realise(pick, { mirrored: false, retro: false }, place, 1);
      step.done = true;
      this.planned += pick.duration;
      const entry = this.entry(pick.name, pick.source, ['as asked', `turns ${Math.round((Math.abs(pick.turn) * 180) / Math.PI)}°`], transforms, phrase, targets.length);
      entry.movement = pick.id;
      return { targets, entry };
    }
    return this.walk(place, { action: 'walk', moves: [{ move: 'turn', angle: 360, distance: null, radius: null }], repeat: 1 }, phrase);
  }

  /** Wave, high reach, or crouch. */
  expressiveAction(place, step, phrase, pose = null) {
    if (step.action === 'crouch' && this.env.seated) return this.standUp(place, step, phrase, false);
    if (step.action === 'crouch') return this.pickUp(place, { ...step, action: 'pick_up', target: 'floor' }, phrase, pose);
    const frames = 48;
    const isSeated = this.env.seated;
    const chair = this.env.chair;
    const target = this.makeDirectTarget(frames, step.action, (local, f, n, c, cache) => {
      const u = n > 1 ? f / (n - 1) : 1;
      const env = Math.sin(u * Math.PI);
      const baseP = isSeated ? chair.seatAnchor : place;
      local.set(this.basePose(baseP, pose, f / SAMPLE_RATE));
      if (isSeated) applySeatedLayer(this.skeleton, local, chair, 1, { leftArmRest: step.action !== 'reach', cache });
      const fx = Math.sin(baseP.heading);
      const fz = Math.cos(baseP.heading);
      const rx = -Math.cos(baseP.heading);
      const rz = Math.sin(baseP.heading);
      const baseH = isSeated ? 1.15 : 1.62;
      if (step.action === 'wave') {
        const sway = Math.sin(u * Math.PI * 5) * 0.14;
        const goal = [baseP.x + rx * (0.28 + sway) + fx * 0.22, baseH, baseP.z + rz * (0.28 + sway) + fz * 0.22];
        applyArmReachIK(this.skeleton, local, goal, env, { hand: 'right', headPitch: 0.08 * env, cache });
      } else {
        // High two-handed reach.
        const goalR = [baseP.x + rx * 0.22 + fx * 0.12, baseH + 0.22, baseP.z + rz * 0.22 + fz * 0.12];
        const goalL = [baseP.x - rx * 0.22 + fx * 0.12, baseH + 0.22, baseP.z - rz * 0.22 + fz * 0.12];
        applyArmReachIK(this.skeleton, local, goalR, env, { hand: 'right', headPitch: 0.20 * env, cache });
        applyArmReachIK(this.skeleton, local, goalL, env, { hand: 'left', cache });
      }
    });
    step.done = true;
    this.planned += frames / SAMPLE_RATE;
    const label = step.action === 'wave' ? 'Waving' : 'Reaching high';
    const entry = this.entry(label, null, ['as asked'], [step.action === 'wave' ? 'gesture with right hand' : 'both arms extended upward'], phrase, 1);
    return { targets: [target], entry };
  }

  /**
   * Copy a partner/video/webcam movement and improvise variations on it
   * ("copy my moves and improvise based on them").
   */
  imitate(place, step, phrase, pose = null) {
    if (this.env.seated) return this.standUp(place, step, phrase, false);
    const imp = this.improviser;
    const partnerPool = imp.movements.filter((m) => (m.tags || []).some((t) => ['partner', 'webcam', 'film'].includes(t)));
    const pool = partnerPool.length ? partnerPool : imp.movements;
    if (!pool.length) return this.stand(place, 1, phrase, pose);

    const k = this.imitateStep++;
    const isCopy = step.mode === 'copy' || (step.mode !== 'riff' && k % 3 === 0);
    let m;
    let v = { mirrored: false, retro: false };
    let tempoScale = 1;
    let opLabel = 'copied faithfully from leader';
    if (isCopy || !this.lastCopied) {
      m = pool[k % pool.length];
      this.lastCopied = m;
    } else {
      m = this.lastCopied;
      const op = ['mirror', 'retrograde', 'faster', 'relative'][(k - 1) % 4];
      if (op === 'mirror') {
        v = { mirrored: true, retro: false };
        opLabel = `improvised variation: mirrored ${m.name}`;
      } else if (op === 'retrograde') {
        v = { mirrored: false, retro: true };
        opLabel = `improvised variation: retrograde of ${m.name}`;
      } else if (op === 'faster') {
        v = { mirrored: true, retro: false };
        tempoScale = 1.25;
        opLabel = `improvised variation: accelerated riff on ${m.name}`;
      } else {
        const rel = imp.relative(m, imp.recent.slice(-2));
        opLabel = `improvised answer to ${m.name} via ${rel.name}`;
        m = rel;
      }
    }
    imp.recent.push(m.id);
    const { targets, transforms } = imp.realise(m, v, place, tempoScale);
    this.planned += m.duration / tempoScale;
    const entry = this.entry(m.name, m.source, ['partner direction', opLabel], transforms, `${phrase} · ${isCopy ? 'copy' : 'improvise'}`, targets.length);
    entry.movement = m.id;
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

  /** Walk (or run) a floor pattern from the dancer's place: one target per gait cycle along it. */
  walk(place, step, phrase, isRun = false) {
    const { walking, standing } = this.material();
    if (!walking) return null;
    const gait = isRun ? { ...walking, speed: walking.speed * 1.75, period: Math.max(0.65, walking.period * 0.72) } : walking;
    const { points, rotation, mirrored } = step.noFitStage
      ? { points: waypoints(place, step, gait), rotation: 0, mirrored: false }
      : fitOnStage(place, step, gait);
    const window = this.window(walking.source, walking.time);
    const a = analysisOf(walking.source, this.skeleton);
    const fr = a.frames[Math.round(walking.time * SAMPLE_RATE)];
    const from = { x: fr.root[0], z: fr.root[2], heading: walking.heading };
    const standWin = standing ? this.window(standing.source, standing.rest) : null;
    const standFrom = standing
      ? (() => {
          const sa = analysisOf(standing.source, this.skeleton);
          const sfr = sa.frames[Math.round(standing.rest * SAMPLE_RATE)];
          return { x: sfr.root[0], z: sfr.root[2], heading: sfr.heading };
        })()
      : null;
    const kind = isRun ? 'run' : 'walk';
    const targets = points.map((p, i) => {
      const prev = i === 0 ? place : points[i - 1];
      const isPivot = Math.hypot(p.x - prev.x, p.z - prev.z) < 0.05;
      const poses = isPivot && standWin
        ? this.place(standWin, standFrom, p.x, p.z, p.heading)
        : this.place(window, from, p.x, p.z, p.heading);
      return { poses, numPoses: POSES_PER_TOKEN, tokens: tokensFor(p.seconds), early: 0, movement: kind };
    });
    step.done = true;
    this.planned += points.reduce((t, p) => t + p.seconds, 0);
    const transforms = [`${points.length} targets along the path`];
    if (rotation) transforms.push(`turned ${rotation}° to stay on stage`);
    if (mirrored) transforms.push('curving the other way to stay on stage');
    const title = step.destLabel
      ? `${isRun ? 'Running' : 'Walking'} to ${step.destLabel}`
      : `${isRun ? 'Running' : 'Walking'} ${describeStep(step).replace(/^(?:walk|run): /, '')}`;
    const entry = this.entry(title, walking.source, ['as asked'], transforms, phrase, targets.length);
    return { targets, entry };
  }
}
