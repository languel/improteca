// Learn movement from a film: MediaPipe Pose on each frame of a video, then
// fit SOMA (src/dancer/fit.js). Runs in the browser; the video must be served
// with CORS (the mediathèque's are).

import { fitClip, fromMediaPipe, LM } from '../src/dancer/fit.js';

const TASKS = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

let landmarkerPromise = null;
async function landmarker() {
  landmarkerPromise ??= (async () => {
    const vision = await import(`${TASKS}/vision_bundle.mjs`);
    const files = await vision.FilesetResolver.forVisionTasks(`${TASKS}/wasm`);
    const options = (delegate) => ({ baseOptions: { modelAssetPath: MODEL, delegate }, runningMode: 'VIDEO', numPoses: 1 });
    try {
      return await vision.PoseLandmarker.createFromOptions(files, options('GPU'));
    } catch {
      return await vision.PoseLandmarker.createFromOptions(files, options('CPU'));
    }
  })();
  return landmarkerPromise;
}

function loadVideo(url) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.crossOrigin = 'anonymous';
    v.muted = true;
    v.playsInline = true;
    v.preload = 'auto';
    v.src = url;
    v.addEventListener('loadeddata', () => resolve(v), { once: true });
    v.addEventListener('error', () => reject(new Error(`Could not load the video ${url}`)), { once: true });
  });
}

function seek(video, t) {
  return new Promise((resolve) => {
    video.addEventListener('seeked', resolve, { once: true });
    video.currentTime = t;
  });
}

/**
 * Track a video and fit SOMA to it.
 * @returns {Promise<{ clip, frames: number, tracked: number }>}
 */
export async function learnFromVideo(url, skeleton, { fps = 15, start = 0, end = Infinity, onProgress = () => {} } = {}) {
  onProgress(0, 'Loading the pose model…');
  const lm = await landmarker();
  const video = await loadVideo(url);
  const stop = Math.min(end, video.duration || 0);
  const world = [];
  const visibility = [];
  const hipImage = [];
  const scale = [];
  let tracked = 0;
  let lastWorld = null;
  let lastVis = null;
  for (let t = start, i = 0; t < stop; t = start + ++i / fps) {
    await seek(video, t);
    const res = lm.detectForVideo(video, Math.round(t * 1000) + 1);
    const w = res.worldLandmarks?.[0];
    const n = res.landmarks?.[0];
    if (w && n) {
      tracked++;
      lastWorld = fromMediaPipe(w);
      lastVis = w.map((p) => p.visibility ?? 1);
      const hipX = (n[LM.hipL].x + n[LM.hipR].x) / 2;
      hipImage.push(hipX);
      // Metres per image width, from the shoulder-to-ankle length in both spaces.
      const img = Math.hypot(((n[LM.shoulderL].x + n[LM.shoulderR].x) - (n[LM.ankleL].x + n[LM.ankleR].x)) / 2 * video.videoWidth,
        ((n[LM.shoulderL].y + n[LM.shoulderR].y) - (n[LM.ankleL].y + n[LM.ankleR].y)) / 2 * video.videoHeight);
      const met = Math.hypot(...[0, 1, 2].map((k) => (lastWorld[LM.shoulderL][k] + lastWorld[LM.shoulderR][k] - lastWorld[LM.ankleL][k] - lastWorld[LM.ankleR][k]) / 2));
      if (img > 1) scale.push((met / img) * video.videoWidth);
    } else {
      hipImage.push(hipImage.length ? hipImage[hipImage.length - 1] : 0.5);
    }
    if (lastWorld) {
      world.push(lastWorld);
      visibility.push(lastVis);
    }
    onProgress((t - start) / (stop - start), `Watching the film… ${Math.round(((t - start) / (stop - start)) * 100)}%`);
  }
  if (tracked < fps) throw new Error('No dancer found in this film.');
  // Sideways travel from the hips' image position (a still camera is assumed).
  scale.sort((a, b) => a - b);
  const metresPerWidth = scale[Math.floor(scale.length / 2)] || 0;
  const hipX = hipImage.slice(hipImage.length - world.length).map((x) => x * metresPerWidth);
  const clip = fitClip({ world, visibility, hipX, fps }, skeleton);
  return { clip, frames: world.length, tracked };
}
