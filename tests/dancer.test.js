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

test('the dancer stays on stage and keeps finding material', { skip }, () => {
  const run = improvise(parsePrompt('travel across the space').intentions, 9, 40);
  for (const c of run) assert.ok(Math.hypot(...c.root) < 4.5, `left the stage at ${c.root}`);
  assert.ok(new Set(run.map((c) => c.entry.movement)).size >= 10, 'stuck on a few movements');
});

// ---- phrase memory -----------------------------------------------------------------

function composed(prompt, seed, steps = 40) {
  const imp = new Improviser({ skeleton: soma, movements: movements(), seed });
  imp.setIntentions(parsePrompt(prompt).intentions);
  let pose = null;
  const entries = [];
  for (let i = 0; i < steps; i++) {
    const { targets, entry } = imp.next(pose);
    pose = targets[targets.length - 1].poses.slice(0, J * TF);
    entries.push(entry);
  }
  return { imp, entries };
}

test('motifs are stated, remembered and developed', { skip }, () => {
  const { imp, entries } = composed('slow and expansive, repetitive', 4);
  const mem = imp.memory();
  assert.ok(mem.motifs.length >= 1 && mem.motifs[0].items.length >= 2);
  const developed = entries.filter((e) => e.reasons[0] && !/^stating|^digression/.test(e.reasons[0]));
  assert.ok(developed.length >= entries.length / 3, `only ${developed.length} of ${entries.length} steps develop a motif`);
  // A developed section dances the motif's own movements.
  const A = imp.composer.motifs[0];
  const againOrAccum = entries.filter((e) => /^(A again|A accumulating|fragment of A|A slower|A faster)/.test(e.phrase || ''));
  for (const e of againOrAccum) assert.ok(A.items.some((it) => it.id === e.movement), `${e.phrase} danced ${e.movement}`);
});

test('development operations transform the motif as named', { skip }, () => {
  // Drive the composer directly, so every operation can be checked.
  const imp = new Improviser({ skeleton: soma, movements: movements(), seed: 2 });
  imp.setIntentions({ variety: 0.5 });
  const c = imp.composer;
  c.next(imp); // establishes the idea
  const ms = movements();
  const motif = { name: 'A', key: c.key, uses: 0, items: [0, 5, 9].map((i) => ({ id: ms[i].id, mirrored: false, retro: false })) };
  c.motifs = [motif];
  c.plan = [];
  const seen = new Set();
  for (let tries = 0; tries < 400 && seen.size < 7; tries++) {
    const plan = c.section(imp, () => true);
    const label = plan[0].phrase;
    // Keep developing A: forget motifs stated by the test loop (never danced here).
    c.motifs = [motif];
    const ids = plan.map((p) => p.item?.id);
    if (label === 'A mirrored') {
      assert.ok(plan.every((p) => p.item.mirrored));
      seen.add('mirror');
    } else if (label === 'A backwards') {
      assert.deepEqual(ids, motif.items.map((it) => it.id).reverse());
      assert.ok(plan.every((p) => p.item.retro));
      seen.add('retrograde');
    } else if (label === 'A accumulating') {
      const m = motif.items.map((it) => it.id);
      assert.deepEqual(ids, [m[0], m[0], m[1], m[0], m[1], m[2]]);
      seen.add('accumulate');
    } else if (label === 'fragment of A') {
      assert.deepEqual(ids, motif.items.slice(0, ids.length).map((it) => it.id));
      assert.ok(ids.length < 3);
      seen.add('fragment');
    } else if (label === 'A varied') {
      assert.equal(plan.filter((p) => p.item.vary).length, 1);
      seen.add('vary');
    } else if (label === 'A slower') {
      assert.ok(plan.every((p) => p.tempo < 1));
      seen.add('slower');
    } else if (label === 'A again') {
      assert.deepEqual(ids, motif.items.map((it) => it.id));
      seen.add('repeat');
    }
  }
  assert.ok(seen.size >= 7, `operations seen: ${[...seen]}`);
});

test('a new prompt starts a new motif; variety sets how much is new', { skip }, () => {
  const imp = new Improviser({ skeleton: soma, movements: movements(), seed: 6 });
  imp.setIntentions(parsePrompt('calm').intentions);
  let pose = null;
  const step = () => {
    const { targets, entry } = imp.next(pose);
    pose = targets[targets.length - 1].poses.slice(0, J * TF);
    return entry;
  };
  for (let i = 0; i < 8; i++) step();
  const before = imp.composer.motifs.length;
  imp.setIntentions(parsePrompt('vigorous ballet').intentions);
  assert.match(step().reasons[0], /^stating /);
  assert.equal(imp.composer.motifs.length, before + 1);

  const share = (prompt) => {
    let stated = 0;
    for (const seed of [1, 2, 3]) stated += composed(prompt, seed, 30).entries.filter((e) => /^stating|^digression/.test(e.reasons[0])).length;
    return stated / 90;
  };
  assert.ok(share('repetitive') + 0.15 < share('unpredictable'), 'variety should raise the share of new material');
});

// ---- video pose fitting ---------------------------------------------------------------

import { fitClip, fromMediaPipe, LM } from '../src/dancer/fit.js';

test('pose fitting recovers limb directions exactly from ideal landmarks', { skip }, () => {
  const file = cmu.find((f) => f.includes('arabesque-05-03')) || cmu[0];
  const { clip } = bvhToSoma(readFileSync(CLIPS + file, 'utf8'), soma, parseBVH);
  // The landmarks a perfect detector would report, in MediaPipe's convention.
  const at = { shoulderL: 'LeftArm', shoulderR: 'RightArm', elbowL: 'LeftForeArm', elbowR: 'RightForeArm', wristL: 'LeftHand', wristR: 'RightHand',
    indexL: 'LeftHandMiddle1', pinkyL: 'LeftHandMiddle1', indexR: 'RightHandMiddle1', pinkyR: 'RightHandMiddle1', hipL: 'LeftLeg', hipR: 'RightLeg',
    kneeL: 'LeftShin', kneeR: 'RightShin', ankleL: 'LeftFoot', ankleR: 'RightFoot', footL: 'LeftToeBase', footR: 'RightToeBase', earL: 'Head', earR: 'Head', nose: 'HeadEnd' };
  const fps = 15;
  const world = [];
  for (let f = 0; f < Math.floor(clip.duration * fps); f++) {
    const w = soma.world(clip.sample(f / fps));
    const hip = [0, 1, 2].map((k) => (w[soma.jointIndex('LeftLeg') * TF + k] + w[soma.jointIndex('RightLeg') * TF + k]) / 2);
    const lm = Array.from({ length: 33 }, () => ({ x: 0, y: 0, z: 0 }));
    for (const [k, joint] of Object.entries(at)) {
      const o = soma.jointIndex(joint) * TF;
      lm[LM[k]] = { x: w[o] - hip[0], y: -(w[o + 1] - hip[1]), z: -(w[o + 2] - hip[2]) };
    }
    world.push(fromMediaPipe(lm));
  }
  const out = fitClip({ world, fps, smoothing: 0 }, soma);
  let worst = 0;
  for (let f = 0; f < world.length; f++) {
    const a = soma.world(clip.sample(f / fps));
    const b = soma.world(out.sample(f / fps));
    for (const [x, y] of [['LeftArm', 'LeftForeArm'], ['RightForeArm', 'RightHand'], ['LeftLeg', 'LeftShin'], ['RightShin', 'RightFoot'], ['LeftFoot', 'LeftToeBase']]) {
      const v = (w) => [0, 1, 2].map((k) => w[soma.jointIndex(y) * TF + k] - w[soma.jointIndex(x) * TF + k]);
      const s = v(a);
      const t = v(b);
      worst = Math.max(worst, Math.acos(Math.min(1, (s[0] * t[0] + s[1] * t[1] + s[2] * t[2]) / Math.hypot(...s) / Math.hypot(...t))));
    }
  }
  assert.ok(worst < 1e-3, `limb direction error ${worst} rad`);
});
