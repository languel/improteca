// Scores: prompts as directed steps (stand, walk a path, dance), and the
// director that turns them into placed MotionBricks targets.
// The director tests need the mediathèque clips in local/clips/ (see DEVELOPING.md).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { bvhToSoma } from '../src/engine/retarget.js';
import { TF } from '../src/engine/math.js';
import { Clip } from '../src/engine/bvh.js';
import { learn } from '../src/dancer/repertoire.js';
import { Improviser } from '../src/dancer/improviser.js';
import { Director, parseScore, sanitizeSteps, mergeSteps, waypoints, fitOnStage, findStanding, findWalk } from '../src/dancer/score.js';
import { sanitize } from '../src/dancer/brains.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const soma = Skeleton.fromJSON(JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8')));
const J = soma.numJoints;
const IDS = ['cmu-everyday-waiting-idle-139-02', 'cmu-everyday-natural-walk-02-01', 'cmu-ballet-arabesque-05-03', 'cmu-dance-side-arabesque-49-09', 'cmu-ballet-grand-jete-05-17', 'cmu-ballet-attitude-cambre-05-18'];
const haveClips = IDS.every((id) => existsSync(`${root}local/clips/${id}.bvh`));
const skip = haveClips ? false : 'mediathèque clips missing in local/clips';

test('prompts with actions become steps; prompts about qualities stay improvisation', () => {
  assert.deepEqual(parseScore('stand in a neutral natural pose then walk in a circle').understood, ['stand', 'walk: arc 360° r 1.5 m']);
  assert.deepEqual(parseScore('walk twice in a small circle clockwise').understood, ['walk: arc right 360° r 1 m, ×2']);
  assert.deepEqual(parseScore('walk 2 m forward, then turn left, then jump; then stand still').understood, ['walk: forward 2 m', 'walk: turn left 90°', 'jump', 'stand']);
  const dance = parseScore('walk forward then dance energetically').steps[1];
  assert.equal(dance.intentions.energy, 0.9);
  for (const p of ['slow and low, like sinking into water', 'calm and still, arms high', 'travel across the space, unpredictable', 'vigorous ballet with jumps and turns']) {
    assert.deepEqual(parseScore(p).steps, [], p);
  }
});

test('model steps are validated', () => {
  const steps = sanitizeSteps([
    { action: 'walk', moves: [{ move: 'forward', distance: 40 }, { move: 'teleport' }, { move: 'turn', angle: 72 }], repeat: 50 },
    { action: 'fly' },
    { action: 'stand', seconds: -3, moves: [{ move: 'forward', distance: 1 }] },
    { action: 'walk' },
  ]);
  assert.equal(steps.length, 3);
  assert.deepEqual(steps[0].moves.map((m) => m.move), ['forward', 'turn']);
  assert.equal(steps[0].moves[0].distance, 6);
  assert.equal(steps[0].repeat, 12);
  assert.equal(steps[1].seconds, 0.5);
  assert.equal(steps[1].moves, null);
  assert.deepEqual(steps[2].moves.map((m) => m.move), ['forward'], 'a walk with no moves walks forward');
  assert.deepEqual(sanitize({ summary: 'x' }).steps, []);
});

const GAIT = { speed: 1.2, period: 1.1 };
const close = (a, b, tol = 1e-6) => Math.hypot(a.x - b.x, a.z - b.z) < tol;

test('moves compose any floor pattern: polygons close, arcs face along the curve', () => {
  const start = { x: 0.5, z: -1, heading: 0.3 };
  // A pentagon, as a brain would compose it: forward, turn 72, five times.
  const pentagon = waypoints(start, { moves: [{ move: 'forward', distance: 1.8 }, { move: 'turn', angle: 72 }], repeat: 5 }, GAIT);
  assert.ok(close(pentagon[pentagon.length - 1], start), 'the pentagon closes');
  // Five edge directions; the last turn is a pivot back to the starting direction.
  const headings = new Set(pentagon.map((p) => (Math.round(((p.heading - start.heading) * 180) / Math.PI) + 360) % 360));
  assert.deepEqual([...headings].sort((a, b) => a - b), [0, 72, 144, 216, 288]);
  // A figure of eight returns to its start and turns both ways.
  const eight = waypoints(start, { moves: [{ move: 'arc', radius: 1, angle: 360 }, { move: 'arc', radius: 1, angle: -360 }] }, GAIT);
  assert.ok(close(eight[eight.length - 1], start));
  // Along an arc, each target faces the way it travels.
  for (let i = 1; i < eight.length; i++) {
    const a = eight[i - 1];
    const b = eight[i];
    const travel = Math.atan2(b.x - a.x, b.z - a.z);
    const mid = (a.heading + b.heading) / 2;
    assert.ok(Math.abs(Math.atan2(Math.sin(travel - mid), Math.cos(travel - mid))) < 0.05);
  }
  // A turn on the spot at the end is a pivot where the walk stopped.
  const pivot = waypoints(start, { moves: [{ move: 'forward', distance: 1 }, { move: 'turn', angle: 180 }] }, GAIT);
  assert.ok(close(pivot[pivot.length - 1], pivot[pivot.length - 2]));
});

test('patterns are turned, as little as needed, to stay on stage', () => {
  const step = { moves: [{ move: 'forward', distance: 4 }] };
  const centre = fitOnStage({ x: 0, z: 0, heading: 0 }, step, GAIT);
  assert.equal(centre.rotation, 0);
  const edge = fitOnStage({ x: 0, z: 2.5, heading: 0 }, step, GAIT);
  assert.notEqual(edge.rotation, 0);
  assert.ok(edge.points.every((p) => Math.hypot(p.x, p.z) <= 3.3));
});

let movementsCache = null;
function movements() {
  if (movementsCache) return movementsCache;
  movementsCache = IDS.flatMap((id) => {
    const { clip } = bvhToSoma(readFileSync(`${root}local/clips/${id}.bvh`, 'utf8'), soma, parseBVH);
    return learn({ id, name: id, clip, tags: [id.split('-')[1]] }, soma);
  });
  return movementsCache;
}

const hipsAndHeading = (poses, p = 0) => {
  const w = soma.world(poses.subarray(p * J * TF, (p + 1) * J * TF));
  const rl = soma.jointIndex('RightLeg');
  const ll = soma.jointIndex('LeftLeg');
  return { x: w[0], z: w[2], heading: Math.atan2(w[rl * TF + 2] - w[ll * TF + 2], -(w[rl * TF] - w[ll * TF])) };
};

test('standing is borrowed from neutral standing, walking from a steady gait', { skip }, () => {
  const sources = [...new Set(movements().map((m) => m.source))];
  assert.equal(findStanding(sources, soma).source.id, 'cmu-everyday-waiting-idle-139-02', 'arms overhead is not neutral standing');
  const walk = findWalk(sources, soma);
  assert.equal(walk.source.id, 'cmu-everyday-natural-walk-02-01');
  assert.ok(walk.speed > 0.9 && walk.speed < 1.6, `walking speed ${walk.speed}`);
  assert.ok(walk.period > 0.8 && walk.period < 1.4, `gait period ${walk.period}`);
});

test('stand, then walk a circle: holds in place, walks around and back, then stands', { skip }, () => {
  const improviser = new Improviser({ skeleton: soma, movements: movements(), seed: 3 });
  const director = new Director({ improviser, skeleton: soma });
  director.setScore(parseScore('stand in a neutral natural pose then walk in a circle').steps, {});
  const all = [];
  let pose = null;
  for (let i = 0; i < 5; i++) {
    const { targets } = director.next(pose);
    all.push(...targets);
    pose = Float64Array.from(targets[targets.length - 1].poses.subarray(3 * J * TF, 4 * J * TF));
  }
  const kinds = all.map((t) => t.movement);
  const firstWalk = kinds.indexOf('walk');
  const lastWalk = kinds.lastIndexOf('walk');
  assert.ok(firstWalk >= 3, 'stands about 4 s first');
  assert.ok(kinds.slice(0, firstWalk).every((k) => k === 'stand'));
  assert.ok(kinds.slice(lastWalk + 1).length >= 3 && kinds.slice(lastWalk + 1).every((k) => k === 'stand'), 'stands after the walk');
  // Holds stay on one spot (the recorded weight shift only).
  const holds = all.slice(0, firstWalk).map((t) => hipsAndHeading(t.poses));
  for (const h of holds) assert.ok(Math.hypot(h.x - holds[0].x, h.z - holds[0].z) < 0.15);
  // The walk goes around a 1.5 m circle and comes back.
  const walk = all.slice(firstWalk, lastWalk + 1).map((t) => hipsAndHeading(t.poses));
  const xs = walk.map((p) => p.x);
  const zs = walk.map((p) => p.z);
  assert.ok(Math.max(...xs) - Math.min(...xs) > 2.5 && Math.max(...zs) - Math.min(...zs) > 2.5, 'about 3 m across');
  const end = walk[walk.length - 1];
  assert.ok(Math.hypot(end.x - holds[0].x, end.z - holds[0].z) < 0.3, 'back where it started');
  // Each walk target faces along the circle (its heading follows the path).
  for (let k = 1; k < walk.length; k++) {
    const d = Math.atan2(Math.sin(walk[k].heading - walk[k - 1].heading), Math.cos(walk[k].heading - walk[k - 1].heading));
    assert.ok(Math.abs(d) > 0.4 && Math.abs(d) < 1.6, `turns steadily, ${d}`);
  }
});

test('a prompt without actions is left to the improviser', { skip }, () => {
  const improviser = new Improviser({ skeleton: soma, movements: movements(), seed: 3 });
  const director = new Director({ improviser, skeleton: soma });
  director.setScore([], { energy: 0.9 });
  const { targets, entry } = director.next(null);
  assert.ok(targets.every((t) => t.movement !== 'stand' && t.movement !== 'walk'));
  assert.ok(entry.name);
});

test("a model's steps win; actions it dropped come back from the built-in reading", () => {
  const model = sanitizeSteps([{ action: 'walk', moves: [{ move: 'forward', distance: 1.8 }, { move: 'turn', angle: 72 }], repeat: 5 }]);
  const builtIn = parseScore('walk in a pentagon shape then jump').steps;
  const { steps, added } = mergeSteps(model, builtIn);
  assert.deepEqual(steps.map((s) => s.action), ['walk', 'jump']);
  assert.equal(steps[0].repeat, 5, "the model's geometry is kept");
  assert.deepEqual(added.map((s) => s.action), ['jump']);
  assert.deepEqual(mergeSteps([], builtIn).steps, builtIn);
  const both = mergeSteps(sanitizeSteps([{ action: 'stand' }, { action: 'walk' }, { action: 'jump' }]), parseScore('stand then walk then jump').steps);
  assert.deepEqual(both.added, []);
});

test('a jump step dances movement that really leaves the floor', { skip }, () => {
  const improviser = new Improviser({ skeleton: soma, movements: movements(), seed: 5 });
  const director = new Director({ improviser, skeleton: soma });
  director.setScore(parseScore('jump').steps, {});
  director.next(null);
  const { targets, entry } = director.next(Float64Array.from(soma.reference));
  const jumps = director.material().jumps;
  // A target is in the air: MotionBricks only leaves the floor when asked to.
  const hips = targets.map((t) => soma.world(t.poses.subarray(0, J * TF))[1]);
  assert.ok(Math.max(...hips) > Math.min(...hips) + 0.15, `targets do not leave the floor: hips ${hips.map((h) => h.toFixed(2))}`);
  assert.ok(jumps.some((j) => j.movement.id === entry.movement), `${entry.name} is not a jump`);
  assert.ok(jumps.every((j) => !/attitude-cambre-05-18 4$/.test(j.movement.name)), 'a held attitude is not a jump');
});

// ---- with Kimodo (a stand-in client: the reference pose carried along the requested path) ----

function fakeKimodo() {
  const requests = [];
  return {
    requests,
    ready: true,
    async generate(req) {
      requests.push(req);
      const frames = Math.round(req.seconds * 30);
      const poses = new Float64Array(frames * J * TF);
      const p = req.path;
      for (let f = 0; f < frames; f++) {
        poses.set(soma.reference, f * J * TF);
        let x = 0;
        let z = 0;
        if (p) {
          // Linear between path points.
          let i = 0;
          while (i < p.frames.length - 1 && p.frames[i + 1] < f) i++;
          const a = p.frames[i];
          const b = p.frames[Math.min(i + 1, p.frames.length - 1)];
          const u = b > a ? Math.min(1, (f - a) / (b - a)) : 0;
          const A = p.xz[i];
          const B = p.xz[Math.min(i + 1, p.xz.length - 1)];
          x = A[0] + (B[0] - A[0]) * u;
          z = A[1] + (B[1] - A[1]) * u;
        }
        poses[f * J * TF] = x;
        poses[f * J * TF + 2] = z;
        if (req.text.includes('jumps')) poses[f * J * TF + 1] += 0.3 * Math.max(0, Math.sin((Math.PI * f) / frames));
      }
      return { clip: new Clip(soma, 30, poses), contacts: [], ms: 5, text: req.text };
    },
  };
}

const settle = () => new Promise((r) => setTimeout(r, 0));

test('with Kimodo, each step is generated from text; walks carry their floor pattern as a path', async () => {
  const kimodo = fakeKimodo();
  const improviser = new Improviser({ skeleton: soma, movements: [], seed: 3 });
  const director = new Director({ improviser, skeleton: soma, kimodo });
  const steps = sanitizeSteps([
    { action: 'stand', seconds: 2 },
    { action: 'walk', moves: [{ move: 'forward', distance: 1.2 }, { move: 'turn', angle: 72 }], repeat: 5 },
    { action: 'jump' },
  ]);
  director.setScore(steps, {}, { prompt: 'stand, walk a pentagon, jump' });
  assert.equal(director.steps.length, 4, 'a score ending in a jump ends standing');
  // Before anything has come back, the dancer holds.
  let pose = Float64Array.from(soma.reference);
  const first = director.next(pose);
  assert.equal(first.targets[0].movement, 'hold');
  await settle();
  assert.deepEqual(kimodo.requests.map((r) => r.text.split(' ').slice(0, 3).join(' ')), ['A person stands', 'A person walks', 'A person jumps', 'A person stands']);
  // The walk asks for its pentagon: five corners, back to the start.
  const walk = kimodo.requests[1];
  assert.ok(walk.path.frames.length >= 6 && walk.seconds <= 10);
  const end = walk.path.xz[walk.path.xz.length - 1];
  assert.ok(Math.hypot(end[0], end[1]) < 0.05, `the pentagon closes: ${end}`);
  const kinds = [];
  for (let i = 0; i < 8; i++) {
    const { targets } = director.next(pose);
    kinds.push(targets[0].movement);
    pose = Float64Array.from(targets[targets.length - 1].poses.subarray(3 * J * TF, 4 * J * TF));
  }
  // Then the final stand holds its last pose until the next prompt.
  assert.deepEqual(kinds.slice(0, 6), ['kimodo:stand', 'kimodo:walk', 'kimodo:jump', 'kimodo:stand', 'hold', 'hold']);
  // A walk longer than one clip is split, each part within Kimodo's 10 s.
  const long = sanitizeSteps([{ action: 'walk', moves: [{ move: 'arc', radius: 2, angle: 720 }] }])[0];
  director.steps = [long];
  const chunks = director.kimodoChunks(long, 0);
  assert.ok(chunks.length >= 3 && chunks.every((c) => c.seconds <= 10), chunks.map((c) => c.seconds).join(', '));
});

test('with Kimodo, a jump keeps its highest moment as a keyframe', async () => {
  const kimodo = fakeKimodo();
  const director = new Director({ improviser: new Improviser({ skeleton: soma, movements: [], seed: 1 }), skeleton: soma, kimodo });
  director.setScore(sanitizeSteps([{ action: 'jump', seconds: 2 }]), {});
  director.next(Float64Array.from(soma.reference));
  await settle();
  const { targets } = director.next(Float64Array.from(soma.reference));
  const hips = targets.map((t) => soma.world(t.poses.subarray(0, J * TF))[1]);
  assert.ok(Math.max(...hips) - soma.world(soma.reference)[1] > 0.28, `top of the jump missed: ${hips.map((h) => h.toFixed(2))}`);
});

test('a prompt without steps becomes an open-ended Kimodo dance that keeps asking for more', async () => {
  const kimodo = fakeKimodo();
  const director = new Director({ improviser: new Improviser({ skeleton: soma, movements: [], seed: 1 }), skeleton: soma, kimodo });
  director.setScore([], {}, { prompt: 'slow and low, like sinking into water' });
  let pose = Float64Array.from(soma.reference);
  for (let i = 0; i < 6; i++) {
    const { targets } = director.next(pose);
    pose = Float64Array.from(targets[targets.length - 1].poses.subarray(3 * J * TF, 4 * J * TF));
    await settle();
  }
  assert.ok(kimodo.requests.length >= 3, `only ${kimodo.requests.length} clips asked for`);
  assert.ok(kimodo.requests.every((r) => r.text.includes('sinking into water')));
  assert.equal(new Set(kimodo.requests.map((r) => JSON.stringify(r.seed))).size, kimodo.requests.length, 'each clip has its own seed');
});
