// Embedded VLM & Biomechanical Movement Critic:
// Observes generated movement segments (both 3D kinematic telemetry via
// audit.js and a multi-frame visual contact sheet captured from the stage
// canvas), evaluates naturalness, coherence, and physical/physiological
// accuracy, and feeds closed-loop guidance & corrections back into the
// constraints, improviser, and director.

import { audit } from '../engine/audit.js';
import { TF } from '../engine/math.js';
import { extractJSON } from './brains.js';
import { mouthPosition } from './environment.js';

export const VLM_PROVIDERS = [
  { id: 'auto', label: 'Auto (local Ollama VLM or active Brain + Biomechanical Critic)' },
  { id: 'ollama-vlm', label: 'Ollama Vision (gemma4 / huihui-4 / d1-omni · localhost:11434)' },
  { id: 'brain', label: 'Active Brain (Gemini / Claude / OpenAI Vision)' },
  { id: 'biomech', label: 'Built-in Biomechanical & Task Critic (instant, offline)' },
];

const PREFERRED_OLLAMA_VLMS = [
  'gemma4:latest',
  'huihui_ai/huihui-4:8b',
  'd1-omni:latest',
  'bonsai-2:27b',
  'mgraffam/qwen3.8-turbo-fusion-heretic:27b',
];

const CRITIC_SYSTEM = `You are an expert movement director, biomechanics analyst, and choreography critic for a real-time 3D virtual performer on the SOMA skeleton.
You are given:
1. The user's prompt / intended action and what movement the performer just executed.
2. Quantitative biomechanical telemetry (foot sliding fraction, 90th-percentile foot slip m/s, 99th-percentile end-effector jerk, knee/elbow hinge error, floor clearance, hip height, and distance to stage props like the table, chair, and glass).
3. When available, a 3-frame visual contact sheet (start, middle, end) from the 3D stage.

Assess the movement and output ONLY a single JSON object (no markdown fences, no extra prose) with these keys:
- "naturalness": number 0 to 10 (how human, fluid, and well-phrased the motion is)
- "coherence": number 0 to 10 (how well it fulfills the prompt / action goal and connects to context)
- "physicsScore": number 0 to 10 (grounded feet, zero slip, pure hinge knees/elbows, anatomical range, smooth handovers)
- "diagnosis": one concise sentence naming what worked and any flaw (e.g. foot slip, transition rush, posture, or incomplete prop reach)
- "guidance": one concise coaching instruction for the next phrase
- "movementDelta": number -1.0 to 1.0 (reward > 0 for smooth, fitting movements; penalty < 0 for jerky or sliding ones)
- "body": object with optional adjustments (0 to 1) for {"feet": number|null, "smooth": number|null, "balance": number|null, "limits": number|null}
- "intentions": object with optional adjustments for {"tempo": number|null, "energy": number|null, "stillness": number|null, "level": number|null}`;

/** Compute quantitative biomechanical and environment-task telemetry for a played segment. */
export function computeTelemetry(skeleton, frames, contacts, { env = null, entry = null, prompt = '' } = {}) {
  const report = audit(skeleton, frames, contacts).summary;
  const n = frames.length;
  const J = skeleton.numJoints;
  const firstW = n ? skeleton.world(frames[0]) : skeleton.world(skeleton.reference);
  const lastW = n ? skeleton.world(frames[n - 1]) : firstW;
  const hipYs = frames.map((f) => skeleton.world(f)[1]);
  const meanHipY = hipYs.length ? hipYs.reduce((a, b) => a + b, 0) / hipYs.length : 1.01;
  const minHipY = hipYs.length ? Math.min(...hipYs) : 1.01;
  const maxHipY = hipYs.length ? Math.max(...hipYs) : 1.01;
  const rootTravel = Math.hypot(lastW[0] - firstW[0], lastW[2] - firstW[2]);

  const hr = skeleton.jointIndex('RightHand') * TF;
  const handPos = [lastW[hr], lastW[hr + 1], lastW[hr + 2]];
  const mouth = n ? mouthPosition(skeleton, frames[n - 1]) : [0, 1.5, 0.15];
  const distToMouth = Math.hypot(handPos[0] - mouth[0], handPos[1] - mouth[1], handPos[2] - mouth[2]);
  const distToTable = env ? Math.hypot(lastW[0] - env.table.standAnchor.x, lastW[2] - env.table.standAnchor.z) : null;
  const distToChair = env ? Math.hypot(lastW[0] - env.chair.seatAnchor.x, lastW[2] - env.chair.seatAnchor.z) : null;
  const distToGlass = env ? Math.hypot(handPos[0] - env.glass.x, handPos[1] - env.glass.y, handPos[2] - env.glass.z) : null;

  return {
    frames: n,
    prompt,
    movement: entry?.movement || '',
    name: entry?.name || '',
    phrase: entry?.phrase || '',
    ...report,
    rootTravel: +rootTravel.toFixed(3),
    meanHipY: +meanHipY.toFixed(3),
    minHipY: +minHipY.toFixed(3),
    maxHipY: +maxHipY.toFixed(3),
    distToMouth: +distToMouth.toFixed(3),
    distToTable: distToTable !== null ? +distToTable.toFixed(3) : null,
    distToChair: distToChair !== null ? +distToChair.toFixed(3) : null,
    distToGlass: distToGlass !== null ? +distToGlass.toFixed(3) : null,
    seated: Boolean(env?.seated),
    holdingGlass: env?.holding === 'glass',
    sips: env?.glass?.sips ?? 0,
  };
}

/**
 * Deterministic biomechanical & task critic: always runs in < 1 ms and provides
 * both a baseline assessment and a fallback when no VLM server is active.
 */
export function evaluateBiomechanics(telemetry, currentBody = { limits: 1, feet: 1, floor: 1, balance: 0, smooth: 1 }, currentIntentions = {}) {
  const t = telemetry;
  const issues = [];
  const positives = [];
  const body = {};
  const intentions = {};

  // 1. Physics & physiology score (0..10)
  let physics = 10;
  const slideFrames = t.slideFrames ?? 0;
  const slideP90 = t.slideP90 ?? 0;
  if (slideFrames > 0.08 || slideP90 > 0.15) {
    physics -= Math.min(3.5, slideFrames * 12 + slideP90 * 4);
    issues.push(`foot slip on ${(slideFrames * 100).toFixed(0)}% of contact frames (${(slideP90 * 100).toFixed(1)} cm/s)`);
    if ((currentBody.feet ?? 1) < 1) body.feet = Math.min(1, +(currentBody.feet + 0.15).toFixed(2));
  } else {
    positives.push('feet well planted');
  }

  if (t.offAxisFrames > 0.01 || t.flexOutFrames > 0.01) {
    physics -= 2.5;
    issues.push('limb hinge or flexion range violation');
    body.limits = 1;
  } else {
    positives.push('pure hinge knees/elbows');
  }

  if (t.belowFloorFrames > 0.01) {
    physics -= 2.0;
    issues.push(`floor penetration (${t.belowFloorMaxCm.toFixed(1)} cm)`);
    body.floor = 1;
  }

  // 2. Naturalness & smoothness score (0..10)
  let naturalness = 9.4;
  if (t.jerkP99 > 1.55) {
    naturalness -= Math.min(3.0, (t.jerkP99 - 1.3) * 1.6);
    issues.push(`sharp velocity change (jerk p99 ${t.jerkP99.toFixed(2)} m/s)`);
    if ((currentBody.smooth ?? 1) < 1) body.smooth = Math.min(1, +(currentBody.smooth + 0.15).toFixed(2));
    const curTempo = currentIntentions.tempo ?? 1;
    if (curTempo > 0.82) intentions.tempo = +Math.max(0.75, curTempo * 0.92).toFixed(2);
  } else {
    positives.push('smooth velocity handovers');
  }

  // 3. Coherence & task fulfillment score (0..10)
  let coherence = 9.2;
  if (t.movement === 'sit') {
    if (t.seated && t.meanHipY < 0.78) positives.push('seated cleanly on the chair');
    else {
      coherence -= 2.5;
      issues.push('pelvis remained too high for a seated posture');
    }
  } else if (t.movement === 'pick_up') {
    if (t.holdingGlass) positives.push('grasped the glass accurately');
    else {
      coherence -= 2.0;
      issues.push('hand did not complete grasp of the glass');
    }
  } else if (t.movement === 'drink') {
    if (t.sips > 0) positives.push('brought glass to mouth and drank');
    else {
      coherence -= 2.0;
      issues.push('drink gesture incomplete');
    }
  } else if (t.prompt && /\b(low|floor|sink|crouch)\b/i.test(t.prompt) && t.meanHipY > 0.96) {
    coherence -= 1.2;
    issues.push('posture stayed high despite low-level prompt');
    intentions.level = 0.1;
  } else if (t.prompt && /\b(calm|slow|still|gentle)\b/i.test(t.prompt) && t.jerkP99 > 1.35) {
    coherence -= 1.0;
    issues.push('movement slightly too restless for a calm prompt');
    intentions.energy = 0.15;
    intentions.stillness = 0.8;
  }

  physics = +Math.max(0, Math.min(10, physics)).toFixed(1);
  naturalness = +Math.max(0, Math.min(10, naturalness)).toFixed(1);
  coherence = +Math.max(0, Math.min(10, coherence)).toFixed(1);

  const avg = (physics + naturalness + coherence) / 3;
  const movementDelta = avg >= 8.8 ? 0.3 : avg < 7.2 ? -0.45 : 0;

  const diagnosis = issues.length
    ? `${t.name || 'Segment'}: ${issues.join('; ')} (${positives.slice(0, 2).join(', ')}).`
    : `${t.name || 'Segment'}: ${positives.join(', ')} (slip ${(slideP90 * 100).toFixed(1)} cm/s, jerk ${t.jerkP99.toFixed(2)}).`;

  const guidance = Object.keys(body).length || Object.keys(intentions).length
    ? `Applied closed-loop correction: ${[
        ...Object.entries(body).map(([k, v]) => `body.${k} → ${Math.round(v * 100)}%`),
        ...Object.entries(intentions).map(([k, v]) => `intentions.${k} → ${v}`),
      ].join(', ')}.`
    : avg >= 8.8
      ? `Endorsed ${t.name || 'movement'} (+${movementDelta.toFixed(2)} repertoire weight).`
      : 'Maintaining current physical constraints and intention weights.';

  return {
    source: 'Biomechanical Critic',
    naturalness,
    coherence,
    physicsScore: physics,
    diagnosis,
    guidance,
    movementDelta,
    body,
    intentions,
    telemetry: t,
  };
}

/** Sanitize and merge a VLM's JSON assessment with the baseline biomechanical assessment. */
export function sanitizeCritique(raw, baseline) {
  const clamp = (v, lo, hi, fallback) => (typeof v === 'number' && Number.isFinite(v) ? +Math.min(hi, Math.max(lo, v)).toFixed(2) : fallback);
  const body = { ...baseline.body };
  if (raw?.body && typeof raw.body === 'object') {
    for (const k of ['limits', 'feet', 'floor', 'balance', 'smooth']) {
      const v = clamp(raw.body[k], 0, 1, null);
      if (v !== null) body[k] = v;
    }
  }
  const intentions = { ...baseline.intentions };
  if (raw?.intentions && typeof raw.intentions === 'object') {
    for (const [k, [lo, hi]] of [['tempo', [0.6, 1.4]], ['energy', [0, 1]], ['stillness', [0, 1]], ['level', [0, 1]], ['size', [0, 1]], ['variety', [0, 1]]]) {
      const v = clamp(raw.intentions[k], lo, hi, null);
      if (v !== null) intentions[k] = v;
    }
  }
  return {
    source: raw?.source || 'VLM Critic',
    naturalness: clamp(raw?.naturalness, 0, 10, baseline.naturalness),
    coherence: clamp(raw?.coherence, 0, 10, baseline.coherence),
    physicsScore: clamp(raw?.physicsScore, 0, 10, baseline.physicsScore),
    diagnosis: typeof raw?.diagnosis === 'string' && raw.diagnosis.trim() ? raw.diagnosis.trim().slice(0, 260) : baseline.diagnosis,
    guidance: typeof raw?.guidance === 'string' && raw.guidance.trim() ? raw.guidance.trim().slice(0, 260) : baseline.guidance,
    movementDelta: clamp(raw?.movementDelta, -1, 1, baseline.movementDelta),
    body,
    intentions,
    telemetry: baseline.telemetry,
  };
}

/**
 * Multi-frame visual contact sheet helper for the browser: stitches up to 3
 * canvas snapshots into a single horizontal filmstrip data URL (JPEG).
 */
export class ContactSheetCapture {
  constructor({ width = 240, height = 180, maxFrames = 3 } = {}) {
    this.width = width;
    this.height = height;
    this.maxFrames = maxFrames;
    this.snapshots = [];
  }

  clear() {
    this.snapshots = [];
  }

  /** Capture one frame from a WebGL canvas element with a short label. */
  capture(sourceCanvas, label = '') {
    if (!sourceCanvas || typeof document === 'undefined') return;
    const frameCanvas = document.createElement('canvas');
    frameCanvas.width = this.width;
    frameCanvas.height = this.height;
    const ctx = frameCanvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(sourceCanvas, 0, 0, this.width, this.height);
    if (label) {
      ctx.fillStyle = 'rgba(15, 14, 13, 0.65)';
      ctx.fillRect(0, this.height - 22, this.width, 22);
      ctx.fillStyle = '#f4efe6';
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(label.slice(0, 34), 6, this.height - 7);
    }
    this.snapshots.push(frameCanvas);
    while (this.snapshots.length > this.maxFrames) this.snapshots.shift();
  }

  /** Composite the captured frames into a single horizontal contact-sheet JPEG data URL. */
  toDataURL(quality = 0.76) {
    if (!this.snapshots.length || typeof document === 'undefined') return null;
    const n = this.snapshots.length;
    const strip = document.createElement('canvas');
    strip.width = this.width * n;
    strip.height = this.height;
    const ctx = strip.getContext('2d');
    if (!ctx) return null;
    this.snapshots.forEach((snap, i) => ctx.drawImage(snap, i * this.width, 0));
    return strip.toDataURL('image/jpeg', quality);
  }
}

export class MovementCritic {
  /**
   * @param {object} opts
   * @param {import('../engine/skeleton.js').Skeleton} opts.skeleton
   * @param {import('../engine/constraints.js').Constraints} [opts.constraints]
   * @param {import('./improviser.js').Improviser} [opts.improviser]
   * @param {import('./score.js').Director} [opts.director]
   * @param {typeof fetch} [opts.fetch]
   */
  constructor({ skeleton, constraints = null, improviser = null, director = null, fetch: f = globalThis.fetch?.bind(globalThis) } = {}) {
    this.skeleton = skeleton;
    this.constraints = constraints;
    this.improviser = improviser;
    this.director = director;
    this.fetch = f;
    this.enabled = true;
    this.autoCorrect = true;
    this.mode = 'auto'; // 'auto' | 'ollama-vlm' | 'brain' | 'biomech'
    this.ollamaUrl = 'http://localhost:11434';
    this.ollamaModel = null; // auto-discovered VLM
    this.history = [];
    this.busy = false;
    this.contactSheet = new ContactSheetCapture();
    this.onAssessment = null; // (assessment) => void
  }

  /** Discover if a local Ollama server has a vision-capable model available. */
  async detectOllamaVLM(timeoutMs = 1200) {
    if (!this.fetch) return null;
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), timeoutMs);
      const res = await this.fetch(`${this.ollamaUrl}/api/tags`, { signal: ctl.signal });
      clearTimeout(timer);
      if (!res.ok) return null;
      const data = await res.json();
      const names = (data?.models ?? []).map((m) => String(m.name || ''));
      for (const pref of PREFERRED_OLLAMA_VLMS) {
        if (names.includes(pref)) {
          this.ollamaModel = pref;
          return pref;
        }
      }
      const anyVision = names.find((n) => /vl|vision|omni|gemma4|huihui|bonsai|qwen3/i.test(n));
      if (anyVision) {
        this.ollamaModel = anyVision;
        return anyVision;
      }
    } catch {
      // Ollama not reachable
    }
    return null;
  }

  /**
   * Assess a completed segment and optionally apply closed-loop feedback to
   * constraints and the improviser.
   */
  async assessSegment({ frames, contacts, entry = null, prompt = '', imageDataUrl = null, brainSettings = null } = {}) {
    const env = this.director?.env || null;
    const curBody = this.constraints?.settings || { limits: 1, feet: 1, floor: 1, balance: 0, smooth: 1 };
    const curIntent = this.improviser?.intentions || {};
    const telemetry = computeTelemetry(this.skeleton, frames, contacts, { env, entry, prompt });
    const baseline = evaluateBiomechanics(telemetry, curBody, curIntent);

    let result = baseline;
    const wantVLM = this.mode !== 'biomech' && !this.busy;
    if (wantVLM) {
      try {
        this.busy = true;
        const vlmRaw = await this.queryVLM({ telemetry, baseline, imageDataUrl, brainSettings });
        if (vlmRaw) result = sanitizeCritique(vlmRaw, baseline);
      } catch {
        // Fall back cleanly to the deterministic biomechanical assessment
      } finally {
        this.busy = false;
      }
    }

    if (imageDataUrl) result.imageDataUrl = imageDataUrl;
    if (this.autoCorrect) this.applyFeedback(result);
    this.history.unshift(result);
    if (this.history.length > 30) this.history.pop();
    this.onAssessment?.(result);
    return result;
  }

  /** Query Ollama VLM or the active Brain with the contact sheet image + telemetry. */
  async queryVLM({ telemetry, baseline, imageDataUrl, brainSettings }) {
    if (!this.fetch) return null;
    const userText = `Prompt / Goal: "${telemetry.prompt || 'free improvisation'}"
Executed movement: "${telemetry.name || telemetry.movement}" (${telemetry.phrase || 'phrase'})
Biomechanical telemetry: ${JSON.stringify(telemetry)}
Baseline audit: ${baseline.diagnosis}
Provide your JSON critique and closed-loop parameter adjustments.`;

    // 1. Try local Ollama VLM when mode is 'auto' or 'ollama-vlm'
    if (this.mode === 'auto' || this.mode === 'ollama-vlm') {
      const model = this.ollamaModel || (await this.detectOllamaVLM(900));
      if (model) {
        const content = imageDataUrl
          ? [
              { type: 'text', text: userText },
              { type: 'image_url', image_url: { url: imageDataUrl } },
            ]
          : userText;
        const res = await this.fetch(`${this.ollamaUrl}/v1/chat/completions`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model,
            messages: [
              { role: 'system', content: CRITIC_SYSTEM },
              { role: 'user', content },
            ],
            max_tokens: 450,
            temperature: 0.2,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          const text = data?.choices?.[0]?.message?.content || '';
          const parsed = extractJSON(text);
          return { ...parsed, source: `VLM (${model})` };
        }
      }
    }

    // 2. Try active remote/local Brain if configured
    if ((this.mode === 'auto' || this.mode === 'brain') && brainSettings && brainSettings.provider && brainSettings.provider !== 'none') {
      const base = String(brainSettings.baseUrl || '').replace(/\/+$/, '');
      if (base && (brainSettings.apiKey || /^http:\/\/(localhost|127\.0\.0\.1)/.test(base))) {
        const headers = { 'Content-Type': 'application/json' };
        if (brainSettings.apiKey) headers.Authorization = `Bearer ${brainSettings.apiKey}`;
        const content = imageDataUrl
          ? [
              { type: 'text', text: userText },
              { type: 'image_url', image_url: { url: imageDataUrl } },
            ]
          : userText;
        const res = await this.fetch(base.endsWith('/chat/completions') ? base : `${base}/chat/completions`, {
          method: 'POST',
          headers,
          body: JSON.stringify({
            model: brainSettings.model,
            messages: [
              { role: 'system', content: CRITIC_SYSTEM },
              { role: 'user', content },
            ],
            max_tokens: 450,
            temperature: 0.2,
          }),
        });
        if (res.ok) {
          const data = await res.json();
          const text = data?.choices?.[0]?.message?.content || '';
          const parsed = extractJSON(text);
          return { ...parsed, source: `VLM (${brainSettings.model})` };
        }
      }
    }
    return null;
  }

  /** Apply the critic's closed-loop adjustments back into Constraints and Improviser. */
  applyFeedback(assessment) {
    if (this.constraints && assessment.body && Object.keys(assessment.body).length) {
      this.constraints.set(assessment.body);
    }
    if (this.improviser) {
      this.improviser.applyFeedback({
        movementId: assessment.telemetry?.movement,
        delta: assessment.movementDelta || 0,
        intentions: assessment.intentions,
      });
    }
  }
}
