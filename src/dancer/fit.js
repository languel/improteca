// Video pose -> SOMA. MediaPipe Pose gives 33 landmarks per frame (3D "world"
// landmarks in metres around the hips); this fits SOMA joint rotations to
// them so a filmed movement can be taught like motion capture.
//
// Method: start each frame from SOMA's rest pose (a T-pose facing +Z, left
// on +X), turn the hips to the frame given by hips and shoulders, the head to
// the frame given by ears and nose, then swing each bone so it points along
// its landmark pair. Bone lengths stay SOMA's. Twist along the limbs is not
// observed and stays at the rest pose's.
//
// Known limits (v1): positions come from one camera, so depth is uncertain;
// the dancer travels only sideways (from the hips' image position); the
// lowest foot is placed on the floor, so jumps lose their height.

import { TF, getTf, setTf, tf, tp, tq, qmul, qinv, qnormalize } from '../engine/math.js';
import { rotationBetween, rotationBetweenFrames } from '../engine/retarget.js';
import { Clip } from '../engine/bvh.js';

// MediaPipe Pose landmark indices.
export const LM = {
  nose: 0, earL: 7, earR: 8, shoulderL: 11, shoulderR: 12, elbowL: 13, elbowR: 14, wristL: 15, wristR: 16,
  pinkyL: 17, pinkyR: 18, indexL: 19, indexR: 20, hipL: 23, hipR: 24, kneeL: 25, kneeR: 26,
  ankleL: 27, ankleR: 28, heelL: 29, heelR: 30, footL: 31, footR: 32,
};

const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

/** SOMA bone -> [child joint, direction from landmarks]. */
function boneTargets(P) {
  const hip = mid(P[LM.hipL], P[LM.hipR]);
  const sh = mid(P[LM.shoulderL], P[LM.shoulderR]);
  const ear = mid(P[LM.earL], P[LM.earR]);
  const handL = mid(P[LM.indexL], P[LM.pinkyL]);
  const handR = mid(P[LM.indexR], P[LM.pinkyR]);
  return [
    ['Spine1', 'Spine2', sub(sh, hip)],
    ['Spine2', 'Chest', sub(sh, hip)],
    ['Chest', 'Neck1', sub(sh, hip)],
    ['Neck1', 'Neck2', sub(ear, sh)],
    ['Neck2', 'Head', sub(ear, sh)],
    ['LeftShoulder', 'LeftArm', sub(P[LM.shoulderL], sh)],
    ['LeftArm', 'LeftForeArm', sub(P[LM.elbowL], P[LM.shoulderL])],
    ['LeftForeArm', 'LeftHand', sub(P[LM.wristL], P[LM.elbowL])],
    ['LeftHand', 'LeftHandMiddle1', sub(handL, P[LM.wristL])],
    ['RightShoulder', 'RightArm', sub(P[LM.shoulderR], sh)],
    ['RightArm', 'RightForeArm', sub(P[LM.elbowR], P[LM.shoulderR])],
    ['RightForeArm', 'RightHand', sub(P[LM.wristR], P[LM.elbowR])],
    ['RightHand', 'RightHandMiddle1', sub(handR, P[LM.wristR])],
    ['LeftLeg', 'LeftShin', sub(P[LM.kneeL], P[LM.hipL])],
    ['LeftShin', 'LeftFoot', sub(P[LM.ankleL], P[LM.kneeL])],
    ['LeftFoot', 'LeftToeBase', sub(P[LM.footL], P[LM.ankleL])],
    ['RightLeg', 'RightShin', sub(P[LM.kneeR], P[LM.hipR])],
    ['RightShin', 'RightFoot', sub(P[LM.ankleR], P[LM.kneeR])],
    ['RightFoot', 'RightToeBase', sub(P[LM.footR], P[LM.ankleR])],
  ];
}

/**
 * Convert MediaPipe world landmarks ({x,y,z} in metres, y down, z away from
 * the camera) to our frame (Y up, +Z toward the camera): a person facing the
 * camera faces +Z with their left on +X, like SOMA's rest pose.
 */
export function fromMediaPipe(landmarks) {
  return landmarks.map((p) => [p.x, -p.y, -p.z]);
}

export class PoseFitter {
  constructor(skeleton) {
    this.sk = skeleton;
    const J = skeleton.numJoints;
    this.J = J;
    this.idx = (n) => skeleton.jointIndex(n);
    this.restWorld = skeleton.world(skeleton.reference);
    const pos = (w, j) => [w[j * TF], w[j * TF + 1], w[j * TF + 2]];
    // Rest-pose frames used for the hips and head.
    const w = this.restWorld;
    this.restUp = sub(pos(w, this.idx('Chest')), pos(w, this.idx('Hips')));
    this.restAcross = sub(pos(w, this.idx('LeftLeg')), pos(w, this.idx('RightLeg')));
    this.restToeY = Math.min(w[this.idx('LeftToeBase') * TF + 1], w[this.idx('RightToeBase') * TF + 1]);
    this.restHipY = w[1];
    this.local = new Float64Array(J * TF);
    this.world = new Float64Array(J * TF);
  }

  /**
   * Fit one frame. P: 33 points in our frame (see fromMediaPipe).
   * Returns a local pose with the hips at (0, rest height, 0); placement and
   * the floor are applied over the whole clip by fitClip().
   */
  fit(P) {
    const sk = this.sk;
    const local = this.local;
    local.set(sk.reference);
    const pos = (j) => [this.world[j * TF], this.world[j * TF + 1], this.world[j * TF + 2]];
    const rotW = (j) => [this.world[j * TF + 3], this.world[j * TF + 4], this.world[j * TF + 5], this.world[j * TF + 6]];
    // Hips: up (hips -> shoulders) and across (right hip -> left hip).
    const hip = mid(P[LM.hipL], P[LM.hipR]);
    const sh = mid(P[LM.shoulderL], P[LM.shoulderR]);
    const q = rotationBetweenFrames(this.restUp, this.restAcross, sub(sh, hip), sub(P[LM.hipL], P[LM.hipR]));
    const h = getTf(local, 0);
    setTf(local, 0, tf([0, this.restHipY, 0], qnormalize(qmul(q, tq(h)))));
    sk.world(local, this.world);
    for (const [name, childName, target] of boneTargets(P)) {
      const j = this.idx(name);
      const k = this.idx(childName);
      if (j < 0 || k < 0) continue;
      const a = pos(j);
      const b = pos(k);
      const r = rotationBetween(sub(b, a), target);
      const parentQ = rotW(sk.parents[j]);
      const l = getTf(local, j);
      setTf(local, j, tf(tp(l), qnormalize(qmul(qinv(parentQ), qmul(r, rotW(j))))));
      sk.world(local, this.world);
    }
    return Float64Array.from(local);
  }
}

/**
 * Smooth landmark tracks with a centred moving average, holding the last
 * confident value where visibility is low.
 * frames: arrays of 33 points (our frame); visibility: arrays of 33 numbers.
 */
export function smoothTracks(frames, visibility = null, radius = 2) {
  const n = frames.length;
  const held = frames.map((f) => f.map((p) => [...p]));
  if (visibility) {
    for (let i = 1; i < n; i++) for (let l = 0; l < 33; l++) if (visibility[i][l] < 0.5) held[i][l] = held[i - 1][l];
  }
  return held.map((_, i) =>
    held[i].map((__, l) => {
      const s = [0, 0, 0];
      let c = 0;
      for (let k = Math.max(0, i - radius); k <= Math.min(n - 1, i + radius); k++) {
        s[0] += held[k][l][0];
        s[1] += held[k][l][1];
        s[2] += held[k][l][2];
        c++;
      }
      return [s[0] / c, s[1] / c, s[2] / c];
    }),
  );
}

/**
 * Fit a whole tracked video into a SOMA clip.
 * @param {object} track
 * @param {number[][][]} track.world  per frame, 33 points in our frame (metres)
 * @param {number[][]} [track.visibility]  per frame, 33 visibilities
 * @param {number[]} [track.hipX]  per frame, sideways hip position in metres (from the image), for travel
 * @param {number} [track.smoothing]  moving-average radius in frames (default 1)
 * @param {number} track.fps
 */
export function fitClip(track, skeleton) {
  const fitter = new PoseFitter(skeleton);
  const J = skeleton.numJoints;
  const smooth = smoothTracks(track.world, track.visibility, track.smoothing ?? 1);
  const n = smooth.length;
  const poses = new Float64Array(n * J * TF);
  for (let f = 0; f < n; f++) poses.set(fitter.fit(smooth[f]), f * J * TF);
  // Floor: lowest toe each frame on the floor, smoothed so it does not jitter.
  const lift = new Float64Array(n);
  const w = new Float64Array(J * TF);
  const toes = ['LeftToeBase', 'RightToeBase', 'LeftFoot', 'RightFoot'].map((x) => skeleton.jointIndex(x));
  for (let f = 0; f < n; f++) {
    skeleton.world(poses.subarray(f * J * TF, (f + 1) * J * TF), w);
    lift[f] = fitter.restToeY - Math.min(...toes.map((j) => w[j * TF + 1]));
  }
  const avg = (arr, i, r) => {
    let s = 0;
    let c = 0;
    for (let k = Math.max(0, i - r); k <= Math.min(arr.length - 1, i + r); k++) {
      s += arr[k];
      c++;
    }
    return s / c;
  };
  for (let f = 0; f < n; f++) {
    const o = f * J * TF;
    poses[o + 1] += avg(lift, f, 3);
    if (track.hipX) poses[o] = avg(track.hipX, f, 2) - track.hipX[0];
  }
  return new Clip(skeleton, track.fps, poses);
}
