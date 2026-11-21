import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Skeleton } from '../src/engine/skeleton.js';
import { Constraints } from '../src/engine/constraints.js';
import { Improviser } from '../src/dancer/improviser.js';
import { Director } from '../src/dancer/score.js';
import { LM } from '../src/dancer/fit.js';
import { detectGesture, pointRayToFloor, PartnerDirector } from '../src/dancer/webcam.js';
import { MovementCritic } from '../src/dancer/critic.js';

const root = fileURLToPath(new URL('..', import.meta.url));
const soma = Skeleton.fromJSON(JSON.parse(readFileSync(`${root}assets/soma/skeleton.json`, 'utf8')));

function neutralLandmarks() {
  const pts = Array.from({ length: 33 }, () => [0, 0, 0]);
  pts[LM.nose] = [0, 0.55, 0.05];
  pts[LM.earL] = [0.08, 0.56, 0];
  pts[LM.earR] = [-0.08, 0.56, 0];
  pts[LM.shoulderL] = [0.18, 0.40, 0];
  pts[LM.shoulderR] = [-0.18, 0.40, 0];
  pts[LM.elbowL] = [0.22, 0.14, 0];
  pts[LM.elbowR] = [-0.22, 0.14, 0];
  pts[LM.wristL] = [0.22, -0.10, 0.05];
  pts[LM.wristR] = [-0.22, -0.10, 0.05];
  pts[LM.indexL] = [0.22, -0.16, 0.07];
  pts[LM.pinkyL] = [0.23, -0.16, 0.05];
  pts[LM.indexR] = [-0.22, -0.16, 0.07];
  pts[LM.pinkyR] = [-0.23, -0.16, 0.05];
  pts[LM.hipL] = [0.10, 0, 0];
  pts[LM.hipR] = [-0.10, 0, 0];
  pts[LM.kneeL] = [0.10, -0.42, 0];
  pts[LM.kneeR] = [-0.10, -0.42, 0];
  pts[LM.ankleL] = [0.10, -0.84, 0];
  pts[LM.ankleR] = [-0.10, -0.84, 0];
  pts[LM.footL] = [0.10, -0.88, 0.12];
  pts[LM.footR] = [-0.10, -0.88, 0.12];
  return pts;
}

test('3D gesture detector recognizes pointing to floor, hands up, crouch, and wave', () => {
  // Neutral pose -> no gesture
  assert.equal(detectGesture(neutralLandmarks()), null);

  // Right arm pointing forward-right toward the floor
  const pointPts = neutralLandmarks();
  pointPts[LM.elbowR] = [-0.40, 0.30, 0.18];
  pointPts[LM.wristR] = [-0.62, 0.20, 0.36];
  const gPoint = detectGesture(pointPts, { x: 0, z: 0 });
  assert.ok(gPoint && gPoint.kind === 'point_to_go');
  assert.equal(gPoint.hand, 'right');
  assert.equal(gPoint.action, 'go_to');
  assert.ok(gPoint.target.x < -0.5 && gPoint.target.z > 0.3, `pointed target: ${JSON.stringify(gPoint.target)}`);

  // Both hands raised above head -> jump
  const jumpPts = neutralLandmarks();
  jumpPts[LM.wristL] = [0.22, 0.80, 0];
  jumpPts[LM.wristR] = [-0.22, 0.80, 0];
  const gJump = detectGesture(jumpPts);
  assert.ok(gJump && gJump.kind === 'hands_up' && gJump.action === 'jump');

  // Deep crouch -> sit
  const crouchPts = neutralLandmarks();
  crouchPts[LM.ankleL] = [0.10, -0.30, 0];
  crouchPts[LM.ankleR] = [-0.10, -0.30, 0];
  const gCrouch = detectGesture(crouchPts);
  assert.ok(gCrouch && gCrouch.kind === 'crouch' && gCrouch.action === 'sit');

  // Direct ray projection helper
  const hit = pointRayToFloor([0, 1.35, 0], [0.5, 0.85, 0.5], { x: 0, z: 0 });
  assert.ok(hit.x > 0.8 && hit.z > 0.8);
});

test('PartnerDirector learns live phrases into the repertoire and fires held gesture commands', () => {
  const improviser = new Improviser({ skeleton: soma, movements: [], seed: 1 });
  const director = new Director({ improviser, skeleton: soma });
  const partner = new PartnerDirector({ skeleton: soma, improviser, director });

  // Feed 30 frames (2 seconds at 15 Hz) of moving partner landmarks -> learns a phrase into repertoire.
  for (let f = 0; f < 30; f++) {
    const pts = neutralLandmarks();
    const wave = Math.sin((f / 30) * Math.PI * 2) * 0.35;
    pts[LM.elbowR] = [-0.35, 0.35 + wave * 0.3, 0.2];
    pts[LM.wristR] = [-0.55, 0.45 + wave * 0.5, 0.3];
    partner.ingestFrame(pts, null, 1000 + f * 66);
  }
  assert.equal(partner.phraseCount, 1);
  assert.ok(improviser.movements.length >= 1, 'learned partner movement');
  assert.ok(improviser.movements[0].tags.includes('partner'));
  assert.equal(director.lastCopied.id, improviser.movements[0].id);

  // Switch to gesture mode and hold a pointing gesture for 4 frames -> fires triggeredGesture.
  partner.mode = 'gesture';
  let fired = null;
  for (let f = 0; f < 5; f++) {
    const pts = neutralLandmarks();
    pts[LM.elbowR] = [-0.40, 0.30, 0.18];
    pts[LM.wristR] = [-0.62, 0.20, 0.36];
    const res = partner.ingestFrame(pts, null, 5000 + f * 66);
    if (res.triggeredGesture) fired = res.triggeredGesture;
  }
  assert.ok(fired && fired.kind === 'point_to_go');
});

test('MovementCritic audits physics, queries VLM with visual contact sheet, and applies closed-loop corrections', async () => {
  const constraints = new Constraints(soma, { limits: 1, feet: 0.6, floor: 1, balance: 0, smooth: 0.6 });
  const improviser = new Improviser({ skeleton: soma, movements: [], seed: 1 });
  const director = new Director({ improviser, skeleton: soma });

  // 1. Clean segment -> high score and positive repertoire endorsement.
  const cleanFrames = Array.from({ length: 20 }, () => Float64Array.from(soma.reference));
  const cleanContacts = Array.from({ length: 20 }, () => new Float32Array([1, 1, 1, 1]));
  const critic = new MovementCritic({ skeleton: soma, constraints, improviser, director, fetch: null });
  critic.mode = 'biomech';
  const cleanRes = await critic.assessSegment({
    frames: cleanFrames,
    contacts: cleanContacts,
    entry: { movement: 'move-clean', name: 'Calm stand' },
    prompt: 'stand still',
  });
  assert.ok(cleanRes.physicsScore >= 9.5);
  assert.ok(cleanRes.naturalness >= 9.0);
  assert.ok(improviser.qualityBias['move-clean'] > 0, 'clean movement endorsed');

  // 2. Segment with simulated foot slip -> critic automatically increases constraints.settings.feet and penalizes movement.
  const slipFrames = Array.from({ length: 20 }, (_, f) => {
    const p = Float64Array.from(soma.reference);
    p[0] = f * 0.015; // 0.45 m/s horizontal root slide while contacts=1
    return p;
  });
  const slipRes = await critic.assessSegment({
    frames: slipFrames,
    contacts: cleanContacts,
    entry: { movement: 'move-slip', name: 'Sliding step' },
    prompt: 'stand still',
  });
  assert.ok(slipRes.physicsScore < 9.0, `physics score dropped on slip: ${slipRes.physicsScore}`);
  assert.ok(constraints.settings.feet > 0.6, `closed-loop raised foot planting strength to ${constraints.settings.feet}`);

  // 3. Multimodal VLM round-trip (Ollama VLM endpoint receiving contact sheet image + telemetry).
  let receivedPayload = null;
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      if (req.url === '/api/tags') {
        return res.end(JSON.stringify({ models: [{ name: 'gemma4:latest' }] }));
      }
      receivedPayload = JSON.parse(body);
      res.end(
        JSON.stringify({
          choices: [
            {
              message: {
                content: JSON.stringify({
                  naturalness: 8.7,
                  coherence: 9.1,
                  physicsScore: 9.4,
                  diagnosis: 'Smooth seated reach; right hand aligned well with the glass.',
                  guidance: 'Slow the drink transition slightly for a calmer sip.',
                  movementDelta: 0.4,
                  body: { feet: 0.95, smooth: 0.9 },
                  intentions: { tempo: 0.88, stillness: 0.65 },
                }),
              },
            },
          ],
        }),
      );
    });
  });
  await new Promise((r) => server.listen(0, r));
  const port = server.address().port;
  try {
    const vlmCritic = new MovementCritic({
      skeleton: soma,
      constraints,
      improviser,
      director,
      fetch: globalThis.fetch.bind(globalThis),
    });
    vlmCritic.ollamaUrl = `http://127.0.0.1:${port}`;
    vlmCritic.mode = 'ollama-vlm';
    const vlmRes = await vlmCritic.assessSegment({
      frames: cleanFrames,
      contacts: cleanContacts,
      entry: { movement: 'move-vlm', name: 'Drink glass' },
      prompt: 'go to the table, sit, pick up the glass and drink',
      imageDataUrl: 'data:image/jpeg;base64,/9j/4AAQSkZJRg==',
    });
    assert.equal(vlmRes.source, 'VLM (gemma4:latest)');
    assert.equal(vlmRes.naturalness, 8.7);
    assert.equal(constraints.settings.feet, 0.95);
    assert.equal(improviser.intentions.tempo, 0.88);
    assert.equal(improviser.qualityBias['move-vlm'], 0.4);
    assert.ok(Array.isArray(receivedPayload.messages[1].content), 'multimodal content array sent');
    assert.equal(receivedPayload.messages[1].content[1].type, 'image_url');
  } finally {
    server.close();
  }
});
