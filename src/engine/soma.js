// The MotionBricks SOMA engine, ported from NVIDIA's runtime
// (motionbricks/model/soma/kernels.py, engine.py, model/generation.py).
//
// generate() takes four context poses (the last four played frames) and one
// to four target poses, all as local joint transforms on the same skeleton
// with the root in world space (Y up, meters, 30 Hz), and returns the motion
// between them. Inference itself is injected (`run`), so the same code runs
// with onnxruntime-web in the browser and onnxruntime-node in tests.

import {
  TF, getTf, setTf, qmul, qinv, qrot, qrotInv, qnormalize, qFromAxisAngle, qFromCols, cross, vnormalize,
  tf, tp, tq, tmul, deltaTransform, applyDelta, rootEndpointDelta, applyRootCorrection, IDENTITY,
} from './math.js';

export const SAMPLE_RATE = 30;
export const POSES_PER_TOKEN = 4;
export const GUIDE_POSES = 2 * POSES_PER_TOKEN;
export const MIN_TOKENS = 6;
export const MAX_TOKENS = 16;
export const BOUNDARY_FRAMES = POSES_PER_TOKEN - 1;
export const MAX_VALID_FRAMES = validFrameCount(MAX_TOKENS);
export const DEFAULT_TOKEN_COUNTS = [8, 9, 10, 11, 12, 13, 14];
export const FOOT_CONTACT_JOINTS = ['LeftFoot', 'LeftToeBase', 'RightFoot', 'RightToeBase'];

export function validFrameCount(tokens) {
  if (tokens < MIN_TOKENS || tokens > MAX_TOKENS) throw new Error(`token count must be ${MIN_TOKENS} to ${MAX_TOKENS}, got ${tokens}`);
  return tokens * POSES_PER_TOKEN - 2 * BOUNDARY_FRAMES;
}

/** Heading from the leg (hip) joint positions, as in training ("hips_pos"). */
function headingFromHips(right, left) {
  let across = [right[0] - left[0], 0, right[2] - left[2]];
  const length = Math.hypot(across[0], across[1], across[2]);
  if (length <= 1e-6) return 0;
  across = [across[0] / length, 0, across[2] / length];
  const forward = cross([0, 1, 0], across);
  return Math.atan2(forward[0], forward[2]);
}

function rotationFrom6d(x, y) {
  const xa = vnormalize(x);
  const za = vnormalize(cross(xa, y));
  const ya = cross(za, xa);
  return qFromCols(xa, ya, za);
}

export class SomaEngine {
  /**
   * @param {object} opts
   * @param {import('./skeleton.js').Skeleton} opts.skeleton  the pose skeleton (root at index 0)
   * @param {string[]} opts.modelJointNames  joint order from the model manifest
   * @param {(inputs: object) => Promise<{pred_global_motions: Float32Array, pred_num_tokens: Int32Array|BigInt64Array}>} opts.run
   */
  constructor({ skeleton, modelJointNames, run }) {
    this.skeleton = skeleton;
    this.modelJointNames = modelJointNames;
    this.run = run;
    const J = skeleton.numJoints;
    this.modelToPose = modelJointNames.map((n) => skeleton.jointIndex(n));
    const missing = modelJointNames.filter((_, i) => this.modelToPose[i] < 0);
    if (missing.length) throw new Error(`The skeleton is missing model joints: ${missing.join(', ')}`);
    if (this.modelToPose[0] !== 0) throw new Error('The skeleton must store the root at index zero.');
    this.poseToModel = new Int32Array(J).fill(-1);
    this.modelToPose.forEach((poseJoint, modelJoint) => {
      const parent = skeleton.parents[poseJoint];
      if (parent === -1 || this.poseToModel[parent] >= 0) this.poseToModel[poseJoint] = modelJoint;
    });
    this.rightLeg = skeleton.jointIndex('RightLeg');
    this.leftLeg = skeleton.jointIndex('LeftLeg');
  }

  /** Pad one to four poses to a four-pose token by repeating the last. */
  token(poses, count) {
    const J = this.skeleton.numJoints;
    const n = count ?? poses.length / (J * TF);
    if (n < 1 || n > POSES_PER_TOKEN) throw new Error('a token holds 1 to 4 poses');
    const out = new Float64Array(POSES_PER_TOKEN * J * TF);
    for (let p = 0; p < POSES_PER_TOKEN; p++) {
      const src = Math.min(p, n - 1);
      out.set(poses.subarray(src * J * TF, (src + 1) * J * TF), p * J * TF);
    }
    return { poses: out, numPoses: n };
  }

  /** Model inputs for a context token and target token (encode_motionbricks_inputs_kernel). */
  encode(context, target, { allowedTokenCounts = DEFAULT_TOKEN_COUNTS, headingAlignment = [1, 0] } = {}) {
    const sk = this.skeleton;
    const J = sk.numJoints;
    const M = this.modelJointNames.length;
    const world = new Float64Array(GUIDE_POSES * J * TF);
    for (let p = 0; p < POSES_PER_TOKEN; p++) {
      sk.world(context.poses, world, IDENTITY, p, p);
      sk.world(target.poses, world, IDENTITY, p, POSES_PER_TOKEN + p);
    }
    const W = (frame, joint) => getTf(world, frame * J + joint);

    // compute_guide_alignment_kernel
    const last = POSES_PER_TOKEN - 1;
    const hips = tp(W(last, 0));
    const current = headingFromHips(tp(W(last, this.rightLeg)), tp(W(last, this.leftLeg)));
    const desired = Math.atan2(headingAlignment[0], headingAlignment[1]);
    const alignment = tf([hips[0], 0, hips[2]], qFromAxisAngle([0, 1, 0], current - desired));
    const aq = tq(alignment);
    const aligned = (pos) => qrotInv(aq, [pos[0] - alignment[0], pos[1] - alignment[1], pos[2] - alignment[2]]);

    const globalRoot = new Float32Array(GUIDE_POSES * 5);
    const localRoot = new Float32Array(GUIDE_POSES * 4);
    const localPoses = new Float32Array(GUIDE_POSES * 267);
    const finalFrame = GUIDE_POSES - 1;
    for (let frame = 0; frame < GUIDE_POSES; frame++) {
      const next = frame === POSES_PER_TOKEN - 1 ? frame : Math.min(frame + 1, finalFrame);
      const curHips = aligned(tp(W(frame, 0)));
      const nextHips = aligned(tp(W(next, 0)));
      const curHeading = headingFromHips(aligned(tp(W(frame, this.rightLeg))), aligned(tp(W(frame, this.leftLeg))));
      const nextHeading = headingFromHips(aligned(tp(W(next, this.rightLeg))), aligned(tp(W(next, this.leftLeg))));
      const dh = Math.atan2(Math.sin(nextHeading - curHeading), Math.cos(nextHeading - curHeading));
      globalRoot.set([curHips[0], curHips[1], curHips[2], Math.cos(curHeading), Math.sin(curHeading)], frame * 5);
      localRoot.set([
        dh * SAMPLE_RATE,
        (nextHips[0] - curHips[0]) * SAMPLE_RATE,
        (nextHips[2] - curHips[2]) * SAMPLE_RATE,
        curHips[1],
      ], frame * 4);
      const base = frame * 267;
      for (let mj = 0; mj < M; mj++) {
        const t = W(frame, this.modelToPose[mj]);
        const jp = aligned(tp(t));
        const jq = qmul(qinv(aq), tq(t));
        if (mj > 0) {
          const o = base + (mj - 1) * 3;
          localPoses[o] = jp[0] - curHips[0];
          localPoses[o + 1] = jp[1];
          localPoses[o + 2] = jp[2] - curHips[2];
        }
        const ro = base + (M - 1) * 3 + mj * 6;
        const xa = qrot(jq, [1, 0, 0]);
        const ya = qrot(jq, [0, 1, 0]);
        localPoses.set([xa[0], xa[1], xa[2], ya[0], ya[1], ya[2]], ro);
      }
    }

    // write_motionbricks_conditions_kernel
    const hasGlobal = new Int32Array(GUIDE_POSES);
    const hasLocalRoot = new Int32Array(GUIDE_POSES);
    const hasPoses = new Int32Array(GUIDE_POSES);
    for (let i = 0; i < GUIDE_POSES; i++) {
      const v = i < POSES_PER_TOKEN + target.numPoses ? 1 : 0;
      hasGlobal[i] = v;
      hasPoses[i] = v;
      hasLocalRoot[i] = i === POSES_PER_TOKEN - 1 || i === POSES_PER_TOKEN + target.numPoses - 1 ? 0 : v;
    }
    const allowed = new Int32Array(MAX_TOKENS - MIN_TOKENS + 1);
    for (const c of allowedTokenCounts) {
      if (c < MIN_TOKENS || c > MAX_TOKENS) throw new Error(`allowed token counts must be ${MIN_TOKENS} to ${MAX_TOKENS}`);
      allowed[c - MIN_TOKENS] = 1;
    }
    if (!allowed.some(Boolean)) throw new Error('allow at least one token count');

    return {
      alignment,
      inputs: {
        global_root_values: globalRoot,
        local_root_values: localRoot,
        local_poses: localPoses,
        has_global_root_values: hasGlobal,
        has_local_root_values: hasLocalRoot,
        has_local_poses: hasPoses,
        allowed_pred_num_tokens: allowed,
      },
    };
  }

  /** Model output -> local poses and foot contacts (decode_motionbricks_output_kernel). */
  decode(pred, predNumTokens, alignment) {
    const sk = this.skeleton;
    const J = sk.numJoints;
    const M = this.modelJointNames.length;
    const width = 366;
    const rawFrames = pred.length / width;
    const rawCount = Math.min(Math.max(predNumTokens * POSES_PER_TOKEN, 0), rawFrames);
    const frameCount = Math.max(0, rawCount - 2 * BOUNDARY_FRAMES);
    const poses = new Float64Array(frameCount * J * TF);
    const contacts = new Float32Array(frameCount * 4);
    const rotOffset = 5 + (M - 1) * 3;
    const readRot = (row, modelJoint) => {
      const o = row + rotOffset + modelJoint * 6;
      return rotationFrom6d([pred[o], pred[o + 1], pred[o + 2]], [pred[o + 3], pred[o + 4], pred[o + 5]]);
    };
    for (let f = 0; f < frameCount; f++) {
      const row = (BOUNDARY_FRAMES + f) * width;
      for (let c = 0; c < 4; c++) contacts[f * 4 + c] = Math.min(Math.max(pred[row + width - 4 + c], 0), 1);
      const hipsRot = readRot(row, 0);
      setTf(poses, f * J, tmul(alignment, tf([pred[row], pred[row + 1], pred[row + 2]], hipsRot)));
      for (let j = 1; j < J; j++) {
        const mj = this.poseToModel[j];
        if (mj < 0) {
          setTf(poses, f * J + j, getTf(sk.reference, j));
          continue;
        }
        const r = readRot(row, mj);
        const pr = readRot(row, this.poseToModel[sk.parents[j]]);
        const ref = getTf(sk.reference, j);
        setTf(poses, f * J + j, tf([ref[0], ref[1], ref[2]], qnormalize(qmul(qinv(pr), r))));
      }
    }
    return { poses, contacts, frameCount };
  }

  /** Endpoint correction in place (compute/apply_endpoint_deltas_kernel). */
  correctEndpoints(out, context, target) {
    const J = this.skeleton.numJoints;
    const n = out.frameCount;
    if (n < 2) return out;
    const lastCtx = POSES_PER_TOKEN - 1;
    const start = [];
    const end = [];
    for (let j = 0; j < J; j++) {
      const first = getTf(out.poses, j);
      const lastGen = getTf(out.poses, (n - 1) * J + j);
      const ctx = getTf(context.poses, lastCtx * J + j);
      const tgt = getTf(target.poses, j);
      if (j === 0) {
        start.push(rootEndpointDelta(first, ctx));
        end.push(rootEndpointDelta(lastGen, tgt));
      } else {
        start.push(deltaTransform(first, ctx));
        end.push(deltaTransform(lastGen, tgt));
      }
    }
    for (let f = 0; f < n; f++) {
      const ew = f / (n - 1);
      const sw = 1 - ew;
      for (let j = 0; j < J; j++) {
        const t = getTf(out.poses, f * J + j);
        if (j === 0) setTf(out.poses, f * J, applyRootCorrection(t, start[0], end[0], ew));
        else setTf(out.poses, f * J + j, applyDelta(applyDelta(t, start[j], sw), end[j], ew));
      }
    }
    return out;
  }

  /**
   * Generate the motion from context to target.
   * @param {{poses: Float64Array, numPoses: number}} context  four poses ending at the current pose
   * @param {{poses: Float64Array, numPoses: number}} target  one to four poses starting at the pose to reach
   */
  async generate(context, target, { allowedTokenCounts = DEFAULT_TOKEN_COUNTS, correct = true, headingAlignment = [1, 0] } = {}) {
    if (context.numPoses !== POSES_PER_TOKEN) throw new Error('the context needs four poses');
    const { alignment, inputs } = this.encode(context, target, { allowedTokenCounts, headingAlignment });
    const t0 = now();
    const result = await this.run(inputs);
    const inferenceMs = now() - t0;
    const tokens = Number(result.pred_num_tokens[0]);
    if (!allowedTokenCounts.includes(tokens)) throw new Error(`model chose ${tokens} tokens, outside ${allowedTokenCounts}`);
    const out = this.decode(result.pred_global_motions, tokens, alignment);
    if (correct) this.correctEndpoints(out, context, target);
    return { ...out, tokens, inferenceMs, sampleRate: SAMPLE_RATE };
  }
}

function now() {
  return globalThis.performance ? performance.now() : Date.now();
}
