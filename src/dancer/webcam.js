// Live webcam & partner mocap direction:
//   1. Gesture recognition in 3D from MediaPipe landmarks (pointing ray to a
//      spot on the stage floor -> point-to-go; hands overhead -> jump/leap;
//      crouch -> sit/low; wave -> wave back).
//   2. Live "copy my moves and improvise based on them": fits SOMA joint
//      rotations to incoming frames in real time, segments phrases into the
//      repertoire tagged 'partner' / 'webcam', and drives the director to copy
//      and riff on them (or mirror live).

import { LM, PoseFitter, fitClip, fromMediaPipe } from './fit.js';
import { learn } from './repertoire.js';
import { POSES_PER_TOKEN, SAMPLE_RATE } from '../engine/soma.js';
import { TF } from '../engine/math.js';

const TASKS = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => {
  const l = len(a);
  return l > 1e-9 ? [a[0] / l, a[1] / l, a[2] / l] : [0, 0, 0];
};
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Project a 3D pointing ray (from shoulder through wrist) onto the stage floor (y = 0)
 * relative to `origin` ({ x, z }), clamped to stage radius.
 */
export function pointRayToFloor(shoulder, wrist, origin = { x: 0, z: 0 }, maxRadius = 2.8) {
  const dir = norm(sub(wrist, shoulder));
  // Shoulder height in world space is about 1.35 m.
  const shoulderY = 1.35;
  const t = dir[1] < -0.12 ? Math.min(3.2, Math.max(0.6, shoulderY / -dir[1])) : 1.85;
  const rawX = (origin.x || 0) + dir[0] * t;
  const rawZ = (origin.z || 0) + dir[2] * t;
  const r = Math.hypot(rawX, rawZ);
  const scale = r > maxRadius ? maxRadius / r : 1;
  return {
    x: +(rawX * scale).toFixed(2),
    z: +(rawZ * scale).toFixed(2),
    direction: dir,
  };
}

/**
 * Detect a deliberate physical gesture from 33 3D landmarks (already converted
 * by fromMediaPipe: Y up, +Z toward camera, +X person's left).
 * @param {number[][]} P  33 [x, y, z] points in metres around the hips
 * @param {{ x?: number, z?: number }} [actorPlace]
 * @returns {{ kind: string, action?: string, hand?: string, target?: object, label: string } | null}
 */
export function detectGesture(P, actorPlace = { x: 0, z: 0 }) {
  if (!Array.isArray(P) || P.length < 33) return null;
  const nose = P[LM.nose];
  const shL = P[LM.shoulderL];
  const shR = P[LM.shoulderR];
  const elL = P[LM.elbowL];
  const elR = P[LM.elbowR];
  const wrL = P[LM.wristL];
  const wrR = P[LM.wristR];
  const hipL = P[LM.hipL];
  const hipR = P[LM.hipR];
  const ankL = P[LM.ankleL];
  const ankR = P[LM.ankleR];

  // 1. Both hands raised high above head -> jump / high energy
  if (wrL[1] > nose[1] + 0.14 && wrR[1] > nose[1] + 0.14) {
    return { kind: 'hands_up', action: 'jump', label: 'Both hands raised → jump' };
  }

  // 2. Deep crouch (hips close to ankles vertically) -> sit / crouch
  const hipY = (hipL[1] + hipR[1]) / 2;
  const ankY = (ankL[1] + ankR[1]) / 2;
  const shY = (shL[1] + shR[1]) / 2;
  const torsoLen = Math.max(0.25, shY - hipY);
  if (hipY - ankY < torsoLen * 0.95) {
    return { kind: 'crouch', action: 'sit', label: 'Crouch gesture → sit' };
  }

  // 3. Pointing gesture: one arm straight and extended outward/forward, the other arm lowered.
  const armStraight = (sh, el, wr) => {
    const u = norm(sub(el, sh));
    const l = norm(sub(wr, el));
    const reach = len(sub(wr, sh));
    const horiz = Math.hypot(wr[0] - sh[0], wr[2] - sh[2]);
    return { straight: dot(u, l) > 0.78 && reach > 0.42 && horiz > 0.34, reach };
  };
  const leftArm = armStraight(shL, elL, wrL);
  const rightArm = armStraight(shR, elR, wrR);
  const leftLow = wrL[1] < shL[1] - 0.12;
  const rightLow = wrR[1] < shR[1] - 0.12;

  if (rightArm.straight && leftLow && wrR[1] < nose[1] + 0.08) {
    const spot = pointRayToFloor(shR, wrR, actorPlace);
    return {
      kind: 'point_to_go',
      action: 'go_to',
      hand: 'right',
      target: { x: spot.x, z: spot.z, label: `pointed (${spot.x}, ${spot.z})` },
      label: `Right arm pointing → go to (${spot.x}, ${spot.z})`,
    };
  }
  if (leftArm.straight && rightLow && wrL[1] < nose[1] + 0.08) {
    const spot = pointRayToFloor(shL, wrL, actorPlace);
    return {
      kind: 'point_to_go',
      action: 'go_to',
      hand: 'left',
      target: { x: spot.x, z: spot.z, label: `pointed (${spot.x}, ${spot.z})` },
      label: `Left arm pointing → go to (${spot.x}, ${spot.z})`,
    };
  }

  // 4. Single hand raised above head -> wave / greeting
  if ((wrR[1] > nose[1] + 0.08 && leftLow) || (wrL[1] > nose[1] + 0.08 && rightLow)) {
    return { kind: 'wave', action: 'wave', label: 'Hand raised → wave' };
  }

  return null;
}

/**
 * Stateful controller for live webcam or partner motion capture direction.
 */
export class PartnerDirector {
  /**
   * @param {object} opts
   * @param {import('../engine/skeleton.js').Skeleton} opts.skeleton
   * @param {import('./improviser.js').Improviser} opts.improviser
   * @param {import('./score.js').Director} opts.director
   */
  constructor({ skeleton, improviser, director }) {
    this.skeleton = skeleton;
    this.improviser = improviser;
    this.director = director;
    this.fitter = new PoseFitter(skeleton);
    this.mode = 'copy_and_riff'; // 'copy_and_riff' | 'gesture' | 'mirror'
    this.fps = 15;
    this.buffer = [];
    this.visBuffer = [];
    this.phraseCount = 0;
    this.gestureHold = { kind: null, count: 0, lastTriggered: 0 };
    this.lastPose = null;
    this.lastGesture = null;
    this.active = false;
    this.stream = null;
    this.video = null;
    this.timer = null;
    this.onUpdate = null; // ({ gesture, phrasesLearned, mode, latestMove }) => void
  }

  /**
   * Ingest one frame of 33 3D points (in SOMA's Y-up frame, from fromMediaPipe).
   * Returns { pose, gesture, learnedMove }.
   */
  ingestFrame(points, visibility = null, nowMs = Date.now()) {
    const pose = this.fitter.fit(points);
    this.lastPose = pose;
    this.buffer.push(points.map((p) => [...p]));
    this.visBuffer.push(visibility ? [...visibility] : Array(33).fill(1));

    const actorPlace = this.improviser.sense(pose).place;
    const gesture = detectGesture(points, actorPlace);
    this.lastGesture = gesture;
    let triggeredGesture = null;

    if (gesture) {
      if (this.gestureHold.kind === gesture.kind) this.gestureHold.count++;
      else this.gestureHold = { kind: gesture.kind, count: 1, lastTriggered: this.gestureHold.lastTriggered };
      // Hold gesture for 4 frames (~0.27 s at 15 Hz) with a 1.8 s cooldown before firing command.
      if (this.mode === 'gesture' && this.gestureHold.count >= 4 && nowMs - this.gestureHold.lastTriggered > 1800) {
        this.gestureHold.lastTriggered = nowMs;
        triggeredGesture = gesture;
      }
    } else {
      this.gestureHold.kind = null;
      this.gestureHold.count = 0;
    }

    let learnedMove = null;
    const phraseFrames = Math.round(this.fps * 2.0); // 2.0 s phrase chunks
    if (this.buffer.length >= phraseFrames) {
      const world = this.buffer.splice(0, phraseFrames);
      const vis = this.visBuffer.splice(0, phraseFrames);
      const clip = fitClip({ world, visibility: vis, fps: this.fps, smoothing: 1 }, this.skeleton);
      const id = `partner-phrase-${++this.phraseCount}`;
      const source = { id, name: `Partner move ${this.phraseCount}`, clip, tags: ['partner', 'webcam'], credit: 'Live partner capture' };
      const moves = learn(source, this.skeleton);
      if (moves.length) {
        this.improviser.addMovements(moves);
        this.director.lastCopied = moves[0];
        learnedMove = moves[0];
      }
    }

    this.onUpdate?.({
      gesture,
      triggeredGesture,
      phrasesLearned: this.phraseCount,
      mode: this.mode,
      latestMove: learnedMove,
    });
    return { pose, gesture, triggeredGesture, learnedMove };
  }

  /** Build a 4-pose mirror target window from the latest tracked partner pose. */
  mirrorTarget(place = { x: 0, z: 0 }) {
    const base = this.lastPose || this.skeleton.reference;
    const mirrored = this.improviser.mirror(base);
    mirrored[0] = place.x;
    mirrored[2] = place.z;
    const J = this.skeleton.numJoints;
    const poses = new Float64Array(POSES_PER_TOKEN * J * TF);
    for (let i = 0; i < POSES_PER_TOKEN; i++) poses.set(mirrored, i * J * TF);
    return { poses, numPoses: POSES_PER_TOKEN, tokens: [6, 7, 8], early: 0, movement: 'partner:mirror' };
  }

  /**
   * Simulate a partner demonstration or pointing gesture from a repertoire clip or synthetic gesture
   * (works without a physical webcam, and for one-click "Copy & Improvise" demos).
   */
  teachPartnerDemo(kind = 'dance') {
    const sk = this.skeleton;
    const at = {
      shoulderL: 'LeftArm', shoulderR: 'RightArm', elbowL: 'LeftForeArm', elbowR: 'RightForeArm',
      wristL: 'LeftHand', wristR: 'RightHand', indexL: 'LeftHandMiddle1', pinkyL: 'LeftHandMiddle1',
      indexR: 'RightHandMiddle1', pinkyR: 'RightHandMiddle1', hipL: 'LeftLeg', hipR: 'RightLeg',
      kneeL: 'LeftShin', kneeR: 'RightShin', ankleL: 'LeftFoot', ankleR: 'RightFoot',
      footL: 'LeftToeBase', footR: 'RightToeBase', earL: 'Head', earR: 'Head', nose: 'HeadEnd',
    };
    const Pick = this.improviser.movements.find((m) => (m.tags || []).includes('dance')) || this.improviser.movements[0];
    const frames = Math.round(this.fps * 2.0);
    let lastOut = null;
    for (let f = 0; f < frames; f++) {
      const t = Pick ? Pick.start + (f / frames) * (Pick.end - Pick.start) : f / this.fps;
      const pose = Pick ? Pick.source.clip.sample(t) : Float64Array.from(sk.reference);
      const w = sk.world(pose);
      const hip = [0, 1, 2].map((k) => (w[sk.jointIndex('LeftLeg') * TF + k] + w[sk.jointIndex('RightLeg') * TF + k]) / 2);
      const pts = Array.from({ length: 33 }, () => [0, 0, 0]);
      for (const [k, joint] of Object.entries(at)) {
        const o = sk.jointIndex(joint) * TF;
        pts[LM[k]] = [w[o] - hip[0], w[o + 1] - hip[1], w[o + 2] - hip[2]];
      }
      if (kind === 'point_table') {
        // Extend right arm pointing toward the table at (1.35, -1.30).
        const shR = pts[LM.shoulderR];
        pts[LM.elbowR] = [shR[0] + 0.22, shR[1] - 0.08, shR[2] - 0.18];
        pts[LM.wristR] = [shR[0] + 0.44, shR[1] - 0.18, shR[2] - 0.36];
        pts[LM.wristL] = [pts[LM.shoulderL][0], pts[LM.shoulderL][1] - 0.45, pts[LM.shoulderL][2]];
      }
      lastOut = this.ingestFrame(pts, null, Date.now() + f * 100);
    }
    return lastOut;
  }

  /** Start live webcam tracking in the browser using MediaPipe Pose Landmarker. */
  async startWebcam(videoElement) {
    if (this.active) return;
    const vision = await import(`${TASKS}/vision_bundle.mjs`);
    const files = await vision.FilesetResolver.forVisionTasks(`${TASKS}/wasm`);
    const options = (delegate) => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numPoses: 1 });
    this.landmarker ??= await vision.PoseLandmarker.createFromOptions(files, options('GPU')).catch(() =>
      vision.PoseLandmarker.createFromOptions(files, options('CPU')),
    );
    this.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360, facingMode: 'user' }, audio: false });
    this.video = videoElement;
    this.video.srcObject = this.stream;
    await this.video.play();
    this.active = true;

    const tick = () => {
      if (!this.active) return;
      if (this.video.readyState >= 2) {
        const res = this.landmarker.detectForVideo(this.video, performance.now());
        const w = res.worldLandmarks?.[0];
        if (w) {
          this.ingestFrame(fromMediaPipe(w), w.map((p) => p.visibility ?? 1), performance.now());
        }
      }
      this.timer = setTimeout(tick, Math.round(1000 / this.fps));
    };
    tick();
  }

  stopWebcam() {
    this.active = false;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
    if (this.stream) {
      for (const t of this.stream.getTracks()) t.stop();
      this.stream = null;
    }
    if (this.video) this.video.srcObject = null;
  }
}
