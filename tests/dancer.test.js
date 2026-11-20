import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { bvhToSoma } from '../src/engine/retarget.js';
import { TF, qnormalize, setTf, getTf, tf } from '../src/engine/math.js';
import { makeMirror, mirrorMap } from '../src/dancer/transforms.js';
import { learn, MIN_LEN } from '../src/dancer/repertoire.js';
import { Improviser, rng } from '../src/dancer/improviser.js';
import { parsePrompt, describeIntentions } from '../src/dancer/intentions.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const soma = Skeleton.fromJSON(JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8')));
const J = soma.numJoints;
const CLIPS = `${root}local/clips/`;
const cmu = existsSync(CLIPS) ? readdirSync(CLIPS).filter((f) => /^cmu-(ballet|dance|everyday-(waiting|direct|reach|natural))/.test(f)) : [];
const skip = cmu.length >= 6 ? false : 'mediathèque CMU clips missing in local/clips';

// ---- prompts ------------------------------------------------------------------------

test('prompts become intentions, with negation and intensity', () => {
  const a = parsePrompt('very energetic ballet with lots of jumps and turns');
  assert.equal(a.intentions.energy, 1);
  assert.ok(a.intentions.jump > 0.9 && a.intentions.turn > 0.9);
  assert.deepEqual(a.intentions.tags, ['ballet']);

  const b = parsePrompt('no ballet, travel across the stage, unpredictable');
  assert.deepEqual(b.intentions.avoid, ['ballet']);
  assert.ok(b.intentions.travel > 0.8, 'negation must not cross the comma');
  assert.equal(b.intentions.variety, 0.9);

  const c = parsePrompt('do not jump. a bit calm, led by the right hand');
  assert.ok(c.intentions.jump < 0.1);
  assert.ok(c.intentions.energy > 0.2 && c.intentions.energy < 0.5);
  assert.equal(c.intentions.focus, 'handR');
  assert.ok(describeIntentions(c.intentions).includes('led by right hand'));

  const d = parsePrompt('dance however you like');
  assert.deepEqual(d.understood, []);
});

// ---- mirror -------------------------------------------------------------------------

test('the mirror reflects every joint exactly onto its counterpart', () => {
  const mirror = makeMirror(soma);
  const map = mirrorMap(soma);
  const random = rng(3);
  for (let trial = 0; trial < 5; trial++) {
    const pose = Float64Array.from(soma.reference);
    setTf(pose, 0, tf([random() - 0.5, 0.9 + random() * 0.2, random() - 0.5], qnormalize([random() - 0.5, random() - 0.5, random() - 0.5, 1])));
    for (let j = 1; j < J; j++) {
      const r = getTf(pose, j);
      setTf(pose, j, tf([r[0], r[1], r[2]], qnormalize([r[3] + (random() - 0.5) * 0.6, r[4] + (random() - 0.5) * 0.6, r[5] + (random() - 0.5) * 0.6, r[6]])));
    }
    const w = soma.world(pose);
    const m = soma.world(mirror(pose));
    let worst = 0;
    for (let j = 0; j < J; j++) {
      const a = j * TF;
      const b = map[j] * TF;
      worst = Math.max(worst, Math.hypot(m[a] + w[b], m[a + 1] - w[b + 1], m[a + 2] - w[b + 2]));
    }
    assert.ok(worst < 1e-4, `mirror position error ${worst} m`);
  }
});

// ---- repertoire and improviser on real material -------------------------------------

let movementsCache = null;
function movements() {
  if (movementsCache) return movementsCache;
  movementsCache = [];
  for (const f of cmu) {
    const id = f.replace('.bvh', '');
    const { clip } = bvhToSoma(readFileSync(CLIPS + f, 'utf8'), soma, parseBVH);
    movementsCache.push(...learn({ id, name: id, clip, tags: [id.split('-')[1]] }, soma));
  }
  return movementsCache;
}

test('clips split into movements of usable length with key moments MotionBricks can reach', { skip }, () => {
  const ms = movements();
  assert.ok(ms.length > 50);
  for (const m of ms) {
    assert.ok(m.duration >= MIN_LEN - 1e-9 && m.duration <= 3.01, `${m.id} lasts ${m.duration}`);
    for (let i = 1; i < m.keys.length; i++) {
      const gap = m.keys[i] - m.keys[i - 1];
      // One MotionBricks generation covers 0.6 to 1.93 s.
      assert.ok(gap >= 0.59 && gap <= 1.93, `${m.id} key gap ${gap}`);
    }
    assert.ok(Number.isFinite(m.energy) && Number.isFinite(m.level) && m.startShape.length === 15);
  }
});

function improvise(intentions, seed, steps = 25) {
  const imp = new Improviser({ skeleton: soma, movements: movements(), seed });
  imp.setIntentions(intentions);
  let pose = null;
  const chosen = [];
  for (let i = 0; i < steps; i++) {
    const { targets, entry } = imp.next(pose);
    const last = targets[targets.length - 1].poses;
    pose = last.slice(0, J * TF); // the body ends at the last key pose
    chosen.push({ entry, m: imp.movements.find((x) => x.id === entry.movement), root: [pose[0], pose[2]] });
  }
  return chosen;
}

test('the same seed and intentions give the same performance', { skip }, () => {
  const a = improvise({ energy: 0.8 }, 11, 10).map((c) => c.entry.movement + c.entry.transforms.join());
  const b = improvise({ energy: 0.8 }, 11, 10).map((c) => c.entry.movement + c.entry.transforms.join());
  const c = improvise({ energy: 0.8 }, 12, 10).map((c) => c.entry.movement + c.entry.transforms.join());
  assert.deepEqual(a, b);
  assert.notDeepEqual(a, c);
});

test('prompts steer the choices', { skip }, () => {
  const mean = (xs) => xs.reduce((s, x) => s + x, 0) / xs.length;
  const calm = improvise(parsePrompt('calm and still').intentions, 5);
  const wild = improvise(parsePrompt('vigorous, jumping and turning').intentions, 5);
  assert.ok(mean(calm.map((c) => c.m.energy)) < 0.6 * mean(wild.map((c) => c.m.energy)), 'energy');
  assert.ok(mean(wild.map((c) => c.m.airborne)) > mean(calm.map((c) => c.m.airborne)) + 0.2, 'jumps');
  const ballet = improvise(parsePrompt('ballet').intentions, 5);
  assert.ok(ballet.filter((c) => c.m.tags.includes('ballet')).length >= 20, 'material');
  const noBallet = improvise(parsePrompt('no ballet').intentions, 5);
  assert.equal(noBallet.filter((c) => c.m.tags.includes('ballet')).length, 0, 'avoid');
});

test('the dancer stays on stage and does not loop one movement', { skip }, () => {
  const run = improvise(parsePrompt('travel across the space').intentions, 9, 40);
  for (const c of run) assert.ok(Math.hypot(...c.root) < 4.5, `left the stage at ${c.root}`);
  for (let i = 1; i < run.length; i++) assert.notEqual(run[i].entry.movement, run[i - 1].entry.movement);
});
