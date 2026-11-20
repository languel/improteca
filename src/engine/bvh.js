// BVH clips, ported from MotionBricks' io/bvh.py and AnimationBuffer.
//
// Positions are converted from centimeters to meters; rotations are applied in
// the order the file lists them. A clip can be remapped onto another skeleton
// by joint name (joints the file lacks keep that skeleton's reference pose).

import { TF, IDENTITY, getTf, setTf, qmul, qnormalize, qFromAxisAngle, tf, tmul, tblend } from './math.js';
import { Skeleton } from './skeleton.js';

export class Clip {
  /**
   * @param {Skeleton} skeleton
   * @param {number} sampleRate  frames per second
   * @param {Float64Array} poses  frames * joints * 7 local transforms
   */
  constructor(skeleton, sampleRate, poses) {
    this.skeleton = skeleton;
    this.sampleRate = sampleRate;
    this.poses = poses;
    this.numFrames = poses.length / (skeleton.numJoints * TF);
  }

  get duration() {
    return Math.max(0, this.numFrames - 1) / this.sampleRate;
  }

  /** Local pose at a time (clamped), blending the two nearest frames; rootOffset is applied to the root. */
  sample(time, out = new Float64Array(this.skeleton.numJoints * TF), rootOffset = IDENTITY, outOffset = 0) {
    const J = this.skeleton.numJoints;
    const last = Math.max(0, this.numFrames - 1);
    const fractional = time * this.sampleRate;
    const frame = Math.min(Math.max(Math.floor(fractional), 0), last);
    let blend = 0;
    if (frame < last) blend = Math.min(Math.max(fractional - frame, 0), 1);
    const next = Math.min(frame + 1, last);
    for (let j = 0; j < J; j++) {
      let t = tblend(getTf(this.poses, frame * J + j), getTf(this.poses, next * J + j), blend);
      if (j === 0) t = tmul(rootOffset, t);
      setTf(out, outOffset + j, t);
    }
    return out;
  }

  /** Remap onto another skeleton by joint name. */
  remap(skeleton) {
    if (skeleton === this.skeleton) return this;
    const J = skeleton.numJoints;
    const src = this.skeleton;
    const mapping = src.names.map((n) => skeleton.jointIndex(n));
    const poses = new Float64Array(this.numFrames * J * TF);
    for (let f = 0; f < this.numFrames; f++) {
      poses.set(skeleton.reference, f * J * TF);
      for (let s = 0; s < src.numJoints; s++) {
        const d = mapping[s];
        if (d >= 0) setTf(poses, f * J + d, getTf(this.poses, f * src.numJoints + s));
      }
    }
    return new Clip(skeleton, this.sampleRate, poses);
  }
}

/**
 * Parse BVH text.
 * @param {string} text
 * @param {{ skeleton?: Skeleton, removeNamespace?: boolean }} [options]
 */
export function parseBVH(text, { skeleton = null, removeNamespace = true } = {}) {
  const lines = text.split(/\r?\n/);
  const names = [];
  const parents = [];
  const offsets = [];
  const channels = [];
  const stack = [];
  let inEndSite = false;
  let i = 0;
  for (; i < lines.length; i++) {
    const tokens = lines[i].trim().split(/\s+/).filter(Boolean);
    if (!tokens.length) continue;
    if (tokens[0] === 'MOTION') break;
    if (tokens[0] === 'ROOT' || tokens[0] === 'JOINT') {
      const name = tokens[1];
      names.push(removeNamespace ? name.split(':').pop() : name);
      parents.push(stack.length ? stack[stack.length - 1] : -1);
      stack.push(names.length - 1);
      offsets.push([0, 0, 0]);
      channels.push([]);
    } else if (tokens[0] === 'End' && tokens[1] === 'Site') {
      inEndSite = true;
    } else if (tokens[0] === 'OFFSET' && !inEndSite) {
      offsets[stack[stack.length - 1]] = tokens.slice(1, 4).map(Number);
    } else if (tokens[0] === 'CHANNELS') {
      const count = Number(tokens[1]);
      const list = tokens.slice(2);
      if (count !== list.length) throw new Error(`BVH joint ${names[stack[stack.length - 1]]} declares ${count} channels but lists ${list.length}`);
      channels[stack[stack.length - 1]] = list;
    } else if (tokens[0] === '}') {
      if (inEndSite) inEndSite = false;
      else stack.pop();
    }
  }
  const reference = [];
  for (const o of offsets) reference.push(o[0] * 0.01, o[1] * 0.01, o[2] * 0.01, 0, 0, 0, 1);
  const fileSkeleton = new Skeleton(names, parents, reference);

  let frameCount = 0;
  let sampleRate = 0;
  for (i++; i < lines.length; i++) {
    const tokens = lines[i].trim().split(/\s+/).filter(Boolean);
    if (!tokens.length) continue;
    if (tokens[0] === 'Frames:') frameCount = Number(tokens[1]);
    else if (tokens[0] === 'Frame' && tokens[1] === 'Time:') {
      sampleRate = Math.round((1 / Number(tokens[2])) * 100) / 100;
      i++;
      break;
    }
  }
  if (!sampleRate) throw new Error('BVH frame time is missing');

  const channelCount = channels.reduce((n, c) => n + c.length, 0);
  const J = names.length;
  const poses = new Float64Array(frameCount * J * TF);
  // Per joint: which columns hold positions and rotations, in file order.
  const plan = [];
  let column = 0;
  for (let j = 0; j < J; j++) {
    const pos = [-1, -1, -1];
    const rot = [];
    for (const ch of channels[j]) {
      const c = ch.toLowerCase();
      if (c.endsWith('position')) pos['xyz'.indexOf(c[0])] = column;
      else if (c.endsWith('rotation')) rot.push({ column, axis: 'xyz'.indexOf(c[0]) });
      column++;
    }
    plan.push({ hasPos: pos.some((c) => c >= 0), pos, rot });
  }

  let frame = 0;
  for (; i < lines.length && frame < frameCount; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const values = line.split(/\s+/).map(Number);
    if (values.length !== channelCount) throw new Error(`BVH frame ${frame} has ${values.length} values, expected ${channelCount}`);
    for (let j = 0; j < J; j++) {
      const p = plan[j];
      const ref = getTf(fileSkeleton.reference, j);
      const position = p.hasPos
        ? p.pos.map((c) => (c >= 0 ? f32(values[c]) * 0.01 : 0))
        : [ref[0], ref[1], ref[2]];
      let q = [ref[3], ref[4], ref[5], ref[6]];
      if (p.rot.length) {
        q = [0, 0, 0, 1];
        for (const r of p.rot) {
          const axis = [0, 0, 0];
          axis[r.axis] = 1;
          q = qmul(q, qnormalize(qFromAxisAngle(axis, (f32(values[r.column]) * Math.PI) / 180)));
        }
        q = qnormalize(q);
      }
      setTf(poses, frame * J + j, tf(position, q));
    }
    frame++;
  }
  if (frame !== frameCount) throw new Error(`BVH has ${frame} frames, header says ${frameCount}`);

  const clip = new Clip(fileSkeleton, sampleRate, poses);
  return skeleton ? clip.remap(skeleton) : clip;
}

// The reference loads motion values as float32.
const f32buf = new Float32Array(1);
function f32(x) {
  f32buf[0] = x;
  return f32buf[0];
}
