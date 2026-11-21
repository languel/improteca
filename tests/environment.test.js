import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { bvhToSoma } from '../src/engine/retarget.js';
import { TF } from '../src/engine/math.js';
import { SomaEngine } from '../src/engine/soma.js';
import { Inbetweener } from '../src/engine/inbetweener.js';
import { Constraints } from '../src/engine/constraints.js';
import { audit } from '../src/engine/audit.js';
import { learn } from '../src/dancer/repertoire.js';
import { Improviser } from '../src/dancer/improviser.js';
import { Director, parseScore } from '../src/dancer/score.js';
import { StageEnvironment, mouthPosition } from '../src/dancer/environment.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const skData = JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8'));
const soma = Skeleton.fromJSON(skData);
const J = soma.numJoints;
const IDS = ['cmu-everyday-waiting-idle-139-02', 'cmu-everyday-natural-walk-02-01', 'cmu-ballet-pirouette-05-02'];
const haveClips = IDS.every((id) => existsSync(`${root}local/clips/${id}.bvh`));
const skip = haveClips ? false : 'mediathèque clips missing in local/clips';

function loadMoves() {
  return IDS.flatMap((id) => {
    const { clip } = bvhToSoma(readFileSync(`${root}local/clips/${id}.bvh`, 'utf8'), soma, parseBVH);
    return learn({ id, name: id, clip, tags: [id.split('-')[1]] }, soma);
  });
}

test('rich actor prompts parse into environment-aware step sequences', () => {
  const s1 = parseScore('go to the table , sit, pick up the glass and drink');
  assert.deepEqual(
    s1.steps.map((s) => [s.action, s.target || null]),
    [
      ['go_to', 'table'],
      ['sit', 'chair'],
      ['pick_up', 'glass'],
      ['drink', 'glass'],
    ],
  );
  assert.deepEqual(s1.understood, ['go to table', 'sit on chair', 'pick up glass', 'drink']);

  const s2 = parseScore('copy my moves and improvise based on them');
  assert.equal(s2.steps.length, 2);
  assert.equal(s2.steps[0].action, 'imitate');
  assert.equal(s2.steps[0].mode, 'copy');
  assert.equal(s2.steps[1].action, 'imitate');
  assert.equal(s2.steps[1].mode, 'riff');

  const s3 = parseScore('run to the table, pick up the glass, drink, put it down, and stand up');
  assert.deepEqual(s3.steps.map((s) => s.action), ['run', 'pick_up', 'drink', 'put_down', 'stand_up']);
});

test('go to the table, sit, pick up the glass and drink executes with seated layering and exact hand IK', { skip }, async () => {
  const improviser = new Improviser({ skeleton: soma, movements: loadMoves(), seed: 7 });
  const env = new StageEnvironment();
  const director = new Director({ improviser, skeleton: soma, env });
  const score = parseScore('go to the table , sit, pick up the glass and drink');
  director.setScore(score.steps, {});

  // Fake runner for pure MotionBricks walk segments so test runs in < 50ms without ONNX runtime overhead.
  const engine = new SomaEngine({
    skeleton: soma,
    modelJointNames: skData.model_joint_names,
    run: async () => ({
      pred_global_motions: new Float32Array(10 * 4 * 366),
      pred_num_tokens: new Int32Array([8]),
    }),
  });
  const player = new Inbetweener(engine, { constraints: new Constraints(soma) });

  let { targets } = director.next(null);
  player.reset(targets[0].poses);
  const pending = targets.slice(1);
  let lastPose = targets[0].poses.slice(3 * J * TF, 4 * J * TF);

  const actionsSeen = [];
  let minHandToGlass = Infinity;
  let minHandToMouth = Infinity;
  let sitHipY = null;
  let drinkHipY = null;
  const allFrames = [];
  const allContacts = [];

  for (let guard = 0; guard < 25; guard++) {
    if (!pending.length) {
      const next = director.next(lastPose);
      pending.push(...next.targets);
    }
    const t = pending.shift();
    actionsSeen.push(t.movement);
    if (t.directPoses) {
      const out = await player.generate(t);
      const count = out.frameCount;
      for (let f = 0; f < count; f++) {
        const pose = out.poses.slice(f * J * TF, (f + 1) * J * TF);
        allFrames.push(pose);
        allContacts.push(out.contacts.slice(f * 4, f * 4 + 4));
        const w = soma.world(pose);
        const hr = soma.jointIndex('RightHand') * TF;
        const hand = [w[hr], w[hr + 1], w[hr + 2]];
        if (t.movement === 'pick_up') {
          const g = env.glass.home;
          minHandToGlass = Math.min(minHandToGlass, Math.hypot(hand[0] - g[0], hand[1] - g[1], hand[2] - g[2]));
        }
        if (t.movement === 'drink') {
          const mouth = mouthPosition(soma, pose);
          minHandToMouth = Math.min(minHandToMouth, Math.hypot(hand[0] - mouth[0], hand[1] - mouth[1], hand[2] - mouth[2]));
          drinkHipY = w[1];
        }
        if (t.movement === 'sit' && f === count - 1) {
          sitHipY = w[1];
        }
      }
      player.commit();
      lastPose = out.poses.slice(player.endFrame * J * TF, (player.endFrame + 1) * J * TF);
    } else {
      lastPose = Float64Array.from(t.poses.subarray(3 * J * TF, 4 * J * TF));
      player.reset(t.poses);
    }
    if (actionsSeen.includes('drink')) break;
  }

  assert.ok(actionsSeen.includes('walk'), 'walked to the table');
  assert.ok(actionsSeen.includes('sit'), 'sat on the chair');
  assert.ok(actionsSeen.includes('pick_up'), 'picked up the glass');
  assert.ok(actionsSeen.includes('drink'), 'drank from the glass');
  assert.equal(env.seated, true, 'actor is seated');
  assert.equal(env.holding, 'glass', 'actor is holding the glass');
  assert.equal(env.glass.sips, 1, 'actor took a drink');
  assert.ok(sitHipY > 0.45 && sitHipY < 0.62, `seated hip height is on the chair seat: ${sitHipY}`);
  assert.ok(drinkHipY > 0.45 && drinkHipY < 0.62, `actor remained seated while drinking: ${drinkHipY}`);
  assert.ok(minHandToGlass < 0.03, `hand reached the glass on the table within ${minHandToGlass.toFixed(4)} m`);
  assert.ok(minHandToMouth < 0.04, `hand brought the glass to the mouth within ${minHandToMouth.toFixed(4)} m`);

  // Verify that subsequent seated holds at the table have no head jerking across frames or segment handovers.
  const seatedHold = director.next(lastPose);
  const headEndIdx = soma.jointIndex('HeadEnd') * TF;
  let prevHeadEnd = null;
  let maxSeatedHeadSpeed = 0;
  for (const t of seatedHold.targets) {
    const out = await player.generate(t);
    for (let f = 1; f <= player.endFrame; f++) {
      const pose = out.poses.slice(f * J * TF, (f + 1) * J * TF);
      const w = soma.world(pose);
      const cur = [w[headEndIdx], w[headEndIdx + 1], w[headEndIdx + 2]];
      if (prevHeadEnd) {
        const speed = Math.hypot(cur[0] - prevHeadEnd[0], cur[1] - prevHeadEnd[1], cur[2] - prevHeadEnd[2]) * 30;
        maxSeatedHeadSpeed = Math.max(maxSeatedHeadSpeed, speed);
      }
      prevHeadEnd = cur;
    }
    player.commit();
  }
  assert.ok(maxSeatedHeadSpeed < 0.03, `head stays calm while seated at the table (max speed ${maxSeatedHeadSpeed.toFixed(4)} m/s)`);

  // Audit the direct 30 Hz interaction frames for physiological & physical accuracy.
  const report = audit(soma, allFrames, allContacts).summary;
  assert.equal(report.offAxisFrames, 0, 'elbows and knees stay pure hinges');
  assert.equal(report.flexOutFrames, 0, 'all joints within anatomical flexion range');
  assert.equal(report.belowFloorFrames, 0, 'nothing below the floor');
});

test('copy my moves and improvise based on them alternates faithful copy and choreographic variations', { skip }, () => {
  const moves = loadMoves();
  moves[0].tags.push('partner');
  const improviser = new Improviser({ skeleton: soma, movements: moves, seed: 3 });
  const director = new Director({ improviser, skeleton: soma });
  director.setScore(parseScore('copy my moves and improvise based on them').steps, {});

  let pose = Float64Array.from(soma.reference);
  const log = [];
  for (let i = 0; i < 6; i++) {
    const { targets, entry } = director.next(pose);
    pose = Float64Array.from(targets[targets.length - 1].poses.subarray(3 * J * TF, 4 * J * TF));
    if (entry && entry.movement !== 'stand') log.push(entry);
  }
  assert.ok(log.some((e) => e.reasons.some((r) => /copied faithfully/.test(r))), 'copies leader movement');
  assert.ok(log.some((e) => e.reasons.some((r) => /improvised/.test(r))), 'improvises variations on leader movement');
});
