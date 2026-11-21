// The Inbetweener loop, ported from NVIDIA's app/inbetweener_demo.py:
// start from four context poses, generate toward the next target, play up to
// its transition point, then use the last four played frames as the next
// context. Each transition builds on the motion that was actually played.

import { TF, getTf, tmul, tf, tblend } from './math.js';
import { POSES_PER_TOKEN, SAMPLE_RATE, DEFAULT_TOKEN_COUNTS, FOOT_CONTACT_JOINTS } from './soma.js';

const CONTACT_MAX_Y = [0.105, 0.052, 0.105, 0.052]; // LeftFoot, LeftToeBase, RightFoot, RightToeBase

/** Estimate 4-channel foot contacts from world height and horizontal speed when a clip has no contact labels. */
export function estimateContacts(skeleton, poses, frameCount) {
  const J = skeleton.numJoints;
  const joints = FOOT_CONTACT_JOINTS.map((n) => skeleton.jointIndex(n));
  const contacts = new Float32Array(frameCount * 4);
  const w = new Float64Array(J * TF);
  let prevW = null;
  for (let f = 0; f < frameCount; f++) {
    skeleton.world(poses.subarray(f * J * TF, (f + 1) * J * TF), w);
    for (let c = 0; c < 4; c++) {
      const o = joints[c] * TF;
      const y = w[o + 1];
      const speed = prevW ? Math.hypot(w[o] - prevW[o], w[o + 2] - prevW[o + 2]) * SAMPLE_RATE : 0;
      contacts[f * 4 + c] = y <= CONTACT_MAX_Y[c] && speed < 0.30 ? 1 : 0;
    }
    prevW = Float64Array.from(w);
  }
  return contacts;
}

/**
 * Sample four poses at 30 Hz from a clip, placed in the world
 * (sample_pose_window in the demo, for a Y-up world).
 * The anchor pose (3 for a start, whose samples end there; 0 for a target,
 * whose samples begin there) has its ground position moved to `place`,
 * keeping its height; the other samples move with it.
 */
export function sampleWindow(clip, time, place, anchor) {
  const J = clip.skeleton.numJoints;
  const poses = new Float64Array(POSES_PER_TOKEN * J * TF);
  const a = clip.sample(time);
  const root = getTf(a, 0);
  const ground = tf([-root[0], 0, -root[2]]);
  const rootOffset = tmul(place, ground);
  for (let f = 0; f < POSES_PER_TOKEN; f++) {
    clip.sample(time + (f - anchor) / SAMPLE_RATE, poses, rootOffset, f * J);
  }
  return poses;
}

/**
 * A target to reach: a pose window plus generation settings.
 * @typedef {object} Target
 * @property {Float64Array} poses  four poses from sampleWindow(..., anchor 0)
 * @property {number} [numPoses]  how many of them inform the model (1 to 4)
 * @property {number[]} [tokens]  allowed token counts
 * @property {number} [early]  transition early by this many frames
 */

export class Inbetweener {
  /** @param {import('./soma.js').SomaEngine} engine */
  /**
   * @param {import('./soma.js').SomaEngine} engine
   * @param {{ constraints?: import('./constraints.js').Constraints }} [options]
   *   constraints correct each played frame, in order, before it plays and
   *   before it becomes the next context.
   */
  constructor(engine, { constraints = null } = {}) {
    this.engine = engine;
    this.constraints = constraints;
    this.first = true;
    this.J = engine.skeleton.numJoints;
    this.context = null;
    this.segment = null;
    this.time = 0;
    this.endFrame = 0;
  }

  /** Start from a four-pose window (sampleWindow(..., anchor 3)). */
  reset(startPoses) {
    this.first = true;
    this.context = this.engine.token(startPoses, POSES_PER_TOKEN);
    this.segment = null;
    this.time = 0;
  }

  /** Generate from the current context toward a target (or stream a 30 Hz direct clip segment with endpoint alignment). */
  async generate(target) {
    const t = this.engine.token(target.poses, target.numPoses ?? POSES_PER_TOKEN);
    let out;
    if (target.directPoses && target.directPoses.length >= POSES_PER_TOKEN * this.J * TF) {
      const poses = Float64Array.from(target.directPoses);
      const frameCount = poses.length / (this.J * TF);
      out = { poses, contacts: null, frameCount, tokens: Math.min(16, Math.max(6, Math.round(frameCount / POSES_PER_TOKEN))), inferenceMs: 0, sampleRate: SAMPLE_RATE };
      if (this.context && target.correct !== false) this.engine.correctEndpoints(out, this.context, t);
      out.contacts = target.directContacts && target.directContacts.length === frameCount * 4
        ? Float32Array.from(target.directContacts)
        : estimateContacts(this.engine.skeleton, out.poses, frameCount);
    } else {
      out = await this.engine.generate(this.context, t, {
        allowedTokenCounts: target.tokens ?? DEFAULT_TOKEN_COUNTS,
      });
    }
    if (typeof target.modifier === 'function') {
      target.modifier(out.poses, out.contacts, out.frameCount, this.engine.skeleton);
    }
    this.segment = out;
    this.time = 0;
    // At least four frames play, because the next context is the last four played frames.
    this.endFrame = Math.max(POSES_PER_TOKEN - 1, out.frameCount - 1 - (target.early ?? 0));
    if (this.constraints) {
      // Frame 0 repeats the last played (already corrected) pose, except at the start.
      const from = this.first ? 0 : 1;
      const act = this.constraints.apply(out.poses, out.contacts, from, this.endFrame);
      out.activity = [];
      act.forEach((a, i) => (out.activity[from + i] = a));
    }
    this.first = false;
    return out;
  }

  get endTime() {
    return this.endFrame / SAMPLE_RATE;
  }

  /** Pose at the current playback time (blending frames). */
  pose(out = new Float64Array(this.J * TF)) {
    const s = this.segment;
    const J = this.J;
    const f = Math.min(this.time * SAMPLE_RATE, this.endFrame);
    const f0 = Math.floor(f);
    const f1 = Math.min(f0 + 1, this.endFrame);
    const b = f - f0;
    for (let j = 0; j < J; j++) {
      const a = getTf(s.poses, f0 * J + j);
      const c = getTf(s.poses, f1 * J + j);
      // Same blend as AnimationBuffer.sample.
      const t = b === 0 ? a : tblend(a, c, b);
      for (let k = 0; k < TF; k++) out[j * TF + k] = t[k];
    }
    return out;
  }

  /** Advance playback. Returns true when the transition point is reached and the context is updated. */
  advance(dt) {
    if (!this.segment) return false;
    this.time = Math.min(this.time + dt, this.endTime);
    if (this.time < this.endTime) return false;
    this.commit();
    return true;
  }

  /** Make the last four played frames the next context. */
  commit() {
    const played = this.endFrame + 1;
    const J = this.J;
    const recent = this.segment.poses.subarray((played - POSES_PER_TOKEN) * J * TF, played * J * TF);
    this.context = this.engine.token(Float64Array.from(recent), POSES_PER_TOKEN);
  }
}

