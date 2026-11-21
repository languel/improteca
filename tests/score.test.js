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
import { learn } from '../src/dancer/repertoire.js';
import { Improviser } from '../src/dancer/improviser.js';
import { Director, parseScore, sanitizeSteps, pathPoint, findStanding, findWalk } from '../src/dancer/score.js';
import { sanitize } from '../src/dancer/brains.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const soma = Skeleton.fromJSON(JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8')));
const J = soma.numJoints;
const IDS = ['cmu-everyday-waiting-idle-139-02', 'cmu-everyday-natural-walk-02-01', 'cmu-ballet-arabesque-05-03', 'cmu-dance-side-arabesque-49-09'];
const haveClips = IDS.every((id) => existsSync(`${root}local/clips/${id}.bvh`));
const skip = haveClips ? false : 'mediathèque clips missing in local/clips';

test('prompts with actions become steps; prompts about qualities stay improvisation', () => {
  assert.deepEqual(parseScore('stand in a neutral natural pose then walk in a circle').understood, ['stand', 'walk a circle (r 1.5 m)']);
  assert.deepEqual(parseScore('walk twice in a small circle clockwise').understood, ['walk a circle to the right (r 1 m) ×2']);
  assert.deepEqual(parseScore('walk 2 m forward, then dance energetically for 5 seconds; then stand still').understood, ['walk 2 m', 'dance 5 s', 'stand']);
  const dance = parseScore('walk forward then dance energetically').steps[1];
  assert.equal(dance.intentions.energy, 0.9);
  for (const p of ['slow and low, like sinking into water', 'calm and still, arms high', 'travel across the space, unpredictable']) {
    assert.deepEqual(parseScore(p).steps, [], p);
  }
});

test('model steps are validated', () => {
  const steps = sanitizeSteps([{ action: 'walk', path: 'spiral', radius: 40 }, { action: 'fly' }, { action: 'stand', seconds: -3 }]);
  assert.equal(steps.length, 2);
  assert.equal(steps[0].path, 'line');
  assert.equal(steps[0].radius, 3);
  assert.equal(steps[1].seconds, 0.5);
  assert.deepEqual(sanitize({ summary: 'x' }).steps, []);
});

test('circle paths close and face along the path', () => {
  const start = { x: 0.5, z: -1, heading: 0.3 };
  for (const direction of ['left', 'right']) {
    const step = { path: 'circle', radius: 1.5, direction };
    const end = pathPoint(start, step, 2 * Math.PI * 1.5);
    assert.ok(Math.hypot(end.x - start.x, end.z - start.z) < 1e-9);
    // Heading is the direction of travel.
    const a = pathPoint(start, step, 1);
    const b = pathPoint(start, step, 1.001);
    const h = Math.atan2(b.x - a.x, b.z - a.z);
    assert.ok(Math.abs(Math.atan2(Math.sin(h - a.heading), Math.cos(h - a.heading))) < 1e-3);
  }
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
