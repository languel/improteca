// FP16 (NVIDIA) versus the 8-bit build (tools/build_int8.py).
//
// The 8-bit model does not reproduce FP16's motion (it picks other, equally
// valid tokens), so the comparison is on what matters: how far the raw model
// output misses each target pose before endpoint correction, along the same
// improvised dance. Needs the local FP16 model, the 8-bit parts in
// models/int8/ and the mediathèque clips; skips otherwise.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { bvhToSoma } from '../src/engine/retarget.js';
import { SomaEngine } from '../src/engine/soma.js';
import { Inbetweener } from '../src/engine/inbetweener.js';
import { ortRunner } from '../src/engine/ort.js';
import { TF } from '../src/engine/math.js';
import { learn } from '../src/dancer/repertoire.js';
import { Improviser } from '../src/dancer/improviser.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const FP16 = `${root}local/models/mmm_v6_soma_fp16.onnx`;
const INT8 = `${root}models/int8/manifest.json`;
const CLIPS = `${root}local/clips/`;
const clips = existsSync(CLIPS) ? readdirSync(CLIPS).filter((f) => /^cmu-(ballet|dance)/.test(f)) : [];
const skip = existsSync(FP16) && existsSync(INT8) && clips.length >= 6 ? false : 'FP16 model, 8-bit parts or clips missing';

function int8Bytes() {
  const manifest = JSON.parse(readFileSync(INT8, 'utf8'));
  const packed = Buffer.concat(manifest.parts.map((p) => readFileSync(`${root}models/int8/${p}`)));
  const raw = gunzipSync(packed);
  assert.equal(createHash('sha256').update(raw).digest('hex'), manifest.sha256, '8-bit parts reassemble to the manifest hash');
  return raw;
}

test('the 8-bit model hits targets as well as FP16', { skip, timeout: 900_000 }, async () => {
  const ort = await import('onnxruntime-node');
  const data = JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8'));
  const sk = Skeleton.fromJSON(data);
  const J = sk.numJoints;
  const movements = [];
  for (const f of clips) {
    const id = f.replace('.bvh', '');
    movements.push(...learn({ id, name: id, clip: bvhToSoma(readFileSync(CLIPS + f, 'utf8'), sk, parseBVH).clip, tags: [] }, sk));
  }
  const joints = ['LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftLeg', 'RightLeg', 'LeftShin', 'RightShin', 'Chest', 'Neck1'].map((n) => sk.jointIndex(n));
  const angle = (a, ao, b, bo) => {
    let d = 0;
    for (let k = 3; k < 7; k++) d += a[ao + k] * b[bo + k];
    const s = d < 0 ? -1 : 1;
    let x = 0;
    for (let k = 3; k < 7; k++) x += (a[ao + k] - s * b[bo + k]) ** 2;
    return 4 * Math.asin(Math.min(1, Math.sqrt(x) / 2));
  };
  const result = {};
  for (const [name, model] of [['fp16', FP16], ['int8', int8Bytes()]]) {
    const session = await ort.InferenceSession.create(model);
    const engine = new SomaEngine({ skeleton: sk, modelJointNames: data.model_joint_names, run: ortRunner(ort, session) });
    const imp = new Improviser({ skeleton: sk, movements, seed: 3 });
    imp.setIntentions({ variety: 0.5 });
    const player = new Inbetweener(engine);
    let pose = null;
    const misses = [];
    const ms = [];
    for (let step = 0; step < 10; step++) {
      const { targets } = imp.next(pose);
      if (!player.context) player.reset(targets[0].poses);
      for (const t of step === 0 ? targets.slice(1) : targets) {
        const tok = engine.token(t.poses, t.numPoses);
        const raw = await engine.generate(player.context, tok, { allowedTokenCounts: t.tokens, correct: false });
        const last = (raw.frameCount - 1) * J * TF;
        for (const j of joints) misses.push(angle(raw.poses, last + j * TF, tok.poses, j * TF));
        const out = await player.generate(t);
        ms.push(out.inferenceMs);
        pose = out.poses.slice(player.endFrame * J * TF, (player.endFrame + 1) * J * TF);
        player.commit();
      }
    }
    const mean = misses.reduce((a, b) => a + b, 0) / misses.length;
    ms.sort((a, b) => a - b);
    result[name] = { meanMissDeg: (mean * 180) / Math.PI, medianMs: ms[Math.floor(ms.length / 2)] };
  }
  console.log(`  FP16: misses targets by ${result.fp16.meanMissDeg.toFixed(1)}° (mean), ${result.fp16.medianMs.toFixed(0)} ms per generation`);
  console.log(`  8-bit: misses targets by ${result.int8.meanMissDeg.toFixed(1)}° (mean), ${result.int8.medianMs.toFixed(0)} ms per generation`);
  assert.ok(result.int8.meanMissDeg < result.fp16.meanMissDeg + 2, '8-bit should hit targets about as well as FP16');
});
