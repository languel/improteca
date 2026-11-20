// The Inbetweener page: pick moments of clips as Start and targets, place
// them on the floor, and play motion generated through them in order. Each
// generation starts from the last four frames of the one before, as in
// NVIDIA's Inbetweener demo. Generation runs one segment ahead of playback.
//
// URL options: ?model=<onnx url> &clips=<bvh url>,<bvh url> &backend=wasm &play

import { Stage, Mannequin } from './stage.js';
import { fetchModel, openSession } from './model.js';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { bvhToSoma } from '../src/engine/retarget.js';
import { SomaEngine, SAMPLE_RATE, POSES_PER_TOKEN, MIN_TOKENS, MAX_TOKENS } from '../src/engine/soma.js';
import { Inbetweener, sampleWindow } from '../src/engine/inbetweener.js';
import { placement, TF } from '../src/engine/math.js';
import { ortRunner, MODEL_URLS } from '../src/engine/ort.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const COLORS = ['#3d7ec8', '#c8553d', '#4e9a6a', '#b3832f', '#8a5cc2', '#c2457f', '#3b9aa3', '#7d8a2e'];
const performerColor = () => getComputedStyle(document.documentElement).getPropertyValue('--performer').trim() || '#2f2c28';
const SAMPLES_BASE = 'https://media.githubusercontent.com/media/NVlabs/motionbricks/main/assets/motions/soma/';
const SAMPLES = [
  'locomotion/idle/idle_loop_001__A268.bvh',
  'locomotion/idle/neutral_looking_around_R_001__A542.bvh',
  'locomotion/bs_1d_linear/Neutral_walk_forward_002__A057.bvh',
  'jump/jump_over_obstacle_0_5m_R_001__A415.bvh',
  'locomotion/idle/warm_up_neck_001__A359.bvh',
  'variations/angry_combat_quickstep_ff_idle_090_R_001__A365.bvh',
];

const MEDIATHEQUE = 'https://languel.github.io/mediatheque/';
const DANCE = ['cmu-everyday-waiting-idle-139-02', 'cmu-ballet-pirouette-05-02', 'cmu-ballet-arabesque-05-03',
  'cmu-dance-arching-arms-49-12', 'cmu-dance-balance-tilt-49-22', 'cmu-ballet-pointe-tendue-05-12',
  'cmu-dance-side-arabesque-49-09', 'cmu-ballet-retire-balance-05-14'];

const state = {
  skeleton: null,
  engine: null,
  backend: '',
  clips: [], // { name, clip }
  poses: [], // { id, clipIndex, time, x, z, yaw, numPoses, tokens:[min,max], early }
  selected: 0,
  nextId: 1,
  playing: false,
  run: 0, // playback run id; a new run invalidates pending generations
  queue: [], // generated segments waiting to play
  current: null, // { poses, endFrame, first }
  time: 0,
  nextTarget: 1,
  generating: false,
  late: 0,
  generations: 0,
  lastMs: 0,
  trail: [],
};

const stage = new Stage($('viewport'));
let performer = null;
const ghosts = [];

// ---- boot -------------------------------------------------------------------

async function boot() {
  const data = await (await fetch('assets/soma/skeleton.json')).json();
  state.skeleton = Skeleton.fromJSON(data);
  state.modelJointNames = data.model_joint_names;
  performer = new Mannequin(stage, state.skeleton, { color: performerColor() });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => performer.setColor(performerColor()));
  performer.setPose(state.skeleton.reference);
  bindUI();
  if (matchMedia('(max-width: 760px)').matches) $('poses-details').open = false;
  renderPoses();
  renderStats();
  requestAnimationFrame(frame);

  const clipUrls = params.get('clips');
  if (clipUrls) await loadClipUrls(clipUrls.split(','));
  if (params.get('model')) await loadModel(params.get('model'));
  if (params.has('play') && state.engine && state.clips.length) play();
}

// ---- model -----------------------------------------------------------------

async function loadModel(source) {
  const status = $('model-status');
  const bar = $('model-progress');
  bar.parentElement.hidden = false;
  $('load-model').disabled = true;
  try {
    let bytes;
    if (source instanceof File) {
      status.textContent = `Reading ${source.name}…`;
      bytes = new Uint8Array(await source.arrayBuffer());
    } else {
      status.textContent = 'Downloading…';
      bytes = await fetchModel(source, (f, note) => {
        bar.style.width = `${(f * 100).toFixed(1)}%`;
        status.textContent = note === 'cached' ? 'Loading from cache…' : `Downloading… ${note}`;
      });
    }
    bar.style.width = '100%';
    status.textContent = 'Starting ONNX Runtime…';
    const { ort, session, backend } = await openSession(bytes, params.get('backend') || 'webgpu');
    state.engine = new SomaEngine({
      skeleton: state.skeleton,
      modelJointNames: state.modelJointNames,
      run: ortRunner(ort, session),
    });
    state.backend = backend;
    status.textContent = `Ready on ${backend === 'webgpu' ? 'WebGPU' : 'WebAssembly (CPU)'}.`;
    window.improteca = { state, ready: true };
  } catch (err) {
    console.error(err);
    status.textContent = `Could not load the model: ${err.message || err}`;
    $('load-model').disabled = false;
  } finally {
    bar.parentElement.hidden = true;
    updateButtons();
  }
}

// ---- clips -----------------------------------------------------------------

function addClip(name, text) {
  const { clip, method } = bvhToSoma(text, state.skeleton, parseBVH);
  state.clips.push({ name, clip });
  const li = document.createElement('li');
  li.textContent = `${name} · ${clip.duration.toFixed(1)} s · ${method}`;
  $('clips').append(li);
  if (state.poses.length === 0) {
    // A first start and target so there is something to play.
    state.poses.push(newPose(0, Math.min(1, clip.duration), 0, 0, 0));
    state.poses.push(newPose(0, Math.min(3, clip.duration), 1.5, -1, 60));
  }
  renderPoses();
  updateButtons();
}

async function loadClipUrls(urls, names = []) {
  for (const [i, url] of urls.entries()) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      addClip(names[i] || url.split('/').pop().replace(/\.bvh$/i, ''), await res.text());
    } catch (err) {
      console.error(err);
      $('model-status').textContent = `Could not load ${url}: ${err.message}`;
    }
  }
}

// ---- poses -----------------------------------------------------------------

function newPose(clipIndex, time, x, z, yaw) {
  return { id: state.nextId++, clipIndex, time, x, z, yaw, numPoses: POSES_PER_TOKEN, tokens: [8, 14], early: 0 };
}

function windowOf(pose, index) {
  const clip = state.clips[pose.clipIndex].clip;
  return sampleWindow(clip, pose.time, placement(pose.x, pose.z, (pose.yaw * Math.PI) / 180), index === 0 ? 3 : 0);
}

function colorOf(index) {
  return index === 0 ? '#6d675d' : COLORS[(index - 1) % COLORS.length];
}

function renderPoses() {
  const list = $('poses');
  list.replaceChildren();
  state.poses.forEach((pose, index) => {
    const clip = state.clips[pose.clipIndex]?.clip;
    const li = document.createElement('li');
    li.className = `pose${index === state.selected ? ' selected' : ''}`;
    li.innerHTML = `
      <div class="head">
        <span class="dot" style="background:${colorOf(index)}"></span>
        <span class="name">${index === 0 ? 'Start' : `Target ${index}`}</span>
        ${index > 0 ? '<button data-act="up" title="Move earlier">↑</button><button data-act="down" title="Move later">↓</button><button data-act="remove" title="Remove">✕</button>' : ''}
      </div>
      <label>Clip <select data-k="clipIndex">${state.clips.map((c, i) => `<option value="${i}"${i === pose.clipIndex ? ' selected' : ''}>${c.name}</option>`).join('')}</select><span></span></label>
      <label>Time <input type="range" data-k="time" min="0" max="${clip ? clip.duration.toFixed(3) : 1}" step="0.0333" value="${pose.time}"><output>${pose.time.toFixed(2)}s</output></label>
      <div class="grid3">
        <label>x <input type="number" data-k="x" step="0.25" value="${pose.x}"></label>
        <label>z <input type="number" data-k="z" step="0.25" value="${pose.z}"></label>
        <label>yaw <input type="number" data-k="yaw" step="15" value="${pose.yaw}"></label>
      </div>
      ${index > 0 ? `<div class="grid3">
        <label title="How many of the four target poses inform the model">poses <input type="number" data-k="numPoses" min="1" max="4" value="${pose.numPoses}"></label>
        <label title="Fewest and most motion tokens (4 frames each)">tokens <input type="number" data-k="tok0" min="${MIN_TOKENS}" max="${MAX_TOKENS}" value="${pose.tokens[0]}"></label>
        <label>to <input type="number" data-k="tok1" min="${MIN_TOKENS}" max="${MAX_TOKENS}" value="${pose.tokens[1]}"></label>
      </div>
      <label title="Move on to the next target this many frames before arriving">early <input type="range" data-k="early" min="0" max="30" step="1" value="${pose.early}"><output>${pose.early}f</output></label>` : ''}`;
    li.addEventListener('pointerdown', () => {
      if (state.selected !== index) {
        state.selected = index;
        list.querySelectorAll('.pose').forEach((el, i) => el.classList.toggle('selected', i === index));
      }
    });
    li.addEventListener('input', (e) => {
      const k = e.target.dataset.k;
      if (!k) return;
      const v = Number(e.target.value);
      if (k === 'tok0') pose.tokens[0] = clamp(v, MIN_TOKENS, pose.tokens[1]);
      else if (k === 'tok1') pose.tokens[1] = clamp(v, pose.tokens[0], MAX_TOKENS);
      else if (k === 'numPoses') pose.numPoses = clamp(Math.round(v), 1, 4);
      else pose[k] = v;
      const out = e.target.parentElement.querySelector('output');
      if (out) out.textContent = k === 'early' ? `${v}f` : `${v.toFixed(2)}s`;
      if (k === 'clipIndex') {
        pose.time = Math.min(pose.time, state.clips[v].clip.duration);
        renderPoses();
      }
      updateGhosts();
    });
    li.addEventListener('click', (e) => {
      const act = e.target.dataset.act;
      if (!act) return;
      if (act === 'remove') state.poses.splice(index, 1);
      if (act === 'up' && index > 1) [state.poses[index - 1], state.poses[index]] = [state.poses[index], state.poses[index - 1]];
      if (act === 'down' && index < state.poses.length - 1) [state.poses[index + 1], state.poses[index]] = [state.poses[index], state.poses[index + 1]];
      state.selected = Math.min(state.selected, state.poses.length - 1);
      renderPoses();
    });
    list.append(li);
  });
  updateGhosts();
  updateButtons();
}

function updateGhosts() {
  while (ghosts.length > state.poses.length) ghosts.pop().dispose();
  state.poses.forEach((pose, index) => {
    if (!ghosts[index]) ghosts[index] = new Mannequin(stage, state.skeleton, { opacity: 0.35 });
    const g = ghosts[index];
    g.setColor(colorOf(index));
    if (!state.clips[pose.clipIndex]) return;
    const w = windowOf(pose, index);
    const J = state.skeleton.numJoints;
    const anchor = index === 0 ? 3 : 0;
    g.setPose(w.subarray(anchor * J * TF, (anchor + 1) * J * TF));
  });
  const show = $('ghosts').checked;
  ghosts.forEach((g) => (g.visible = show));
}

function randomTargets(count = 6) {
  const start = state.poses[0];
  state.poses = [start];
  let x = start.x;
  let z = start.z;
  let yaw = start.yaw;
  for (let i = 0; i < count; i++) {
    const clipIndex = Math.floor(Math.random() * state.clips.length);
    const clip = state.clips[clipIndex].clip;
    yaw += (Math.random() - 0.5) * 180;
    const step = 0.6 + Math.random() * 1.4;
    x = clamp(x + Math.sin((yaw * Math.PI) / 180) * step, -4, 4);
    z = clamp(z + Math.cos((yaw * Math.PI) / 180) * step, -4, 4);
    const pose = newPose(clipIndex, Math.random() * Math.max(0, clip.duration - 0.2), round(x), round(z), Math.round(yaw / 15) * 15);
    pose.numPoses = 1 + Math.floor(Math.random() * 4);
    state.poses.push(pose);
  }
  renderPoses();
}

// ---- playback ----------------------------------------------------------------

function play() {
  if (!state.engine || state.poses.length < 2) return;
  stop();
  state.run++;
  state.playing = true;
  state.player = new Inbetweener(state.engine);
  state.player.reset(windowOf(state.poses[0], 0));
  state.windows = state.poses.map((p, i) => windowOf(p, i));
  state.targets = state.poses.slice(1);
  state.nextTarget = 0;
  state.queue = [];
  state.current = null;
  state.time = 0;
  state.late = 0;
  state.trail = [];
  pump(state.run);
  updateButtons();
}

function stop() {
  state.playing = false;
  state.run++;
  state.queue = [];
  state.current = null;
  updateButtons();
}

/** Keep one segment generated ahead of the one playing. */
async function pump(run) {
  if (state.generating) return;
  while (run === state.run && state.playing && state.queue.length < 2) {
    if (state.nextTarget >= state.targets.length) {
      if (!$('loop').checked) break;
      state.nextTarget = 0; // loop back through the targets from where the motion is
    }
    const t = state.targets[state.nextTarget];
    const index = state.nextTarget + 1;
    state.generating = true;
    let out;
    try {
      out = await state.player.generate({
        poses: state.windows[index],
        numPoses: t.numPoses,
        tokens: range(t.tokens[0], t.tokens[1]),
        early: t.early,
      });
    } catch (err) {
      console.error(err);
      $('model-status').textContent = `Generation failed: ${err.message || err}`;
      state.generating = false;
      stop();
      return;
    } finally {
      state.generating = false;
    }
    if (run !== state.run) return;
    // The played frames of this segment are known now, so the next context is too.
    state.player.commit();
    state.queue.push({ poses: out.poses, endFrame: state.player.endFrame, tokens: out.tokens, target: index });
    state.generations++;
    state.lastMs = out.inferenceMs;
    state.nextTarget++;
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (state.playing) advance(dt);
  if (state.playing && $('follow').checked) stage.follow(performer.root());
  stage.render();
  if (Math.floor(now / 250) !== Math.floor((now - dt * 1000) / 250)) renderStats();
  requestAnimationFrame(frame);
}

function advance(dt) {
  const J = state.skeleton.numJoints;
  if (!state.current) {
    if (!state.queue.length) {
      if (!state.generating && state.nextTarget >= state.targets.length && !$('loop').checked) stop();
      return;
    }
    state.current = state.queue.shift();
    state.time = state.trail.length ? 1 / SAMPLE_RATE : 0; // frame 0 repeats the last played pose
    pump(state.run);
  }
  const seg = state.current;
  const end = seg.endFrame / SAMPLE_RATE;
  state.time = Math.min(state.time + dt, end);
  const f = state.time * SAMPLE_RATE;
  const f0 = Math.floor(f);
  performer.setPose(blendFrames(seg.poses, f0, Math.min(f0 + 1, seg.endFrame), f - f0, J));
  const r = performer.root();
  state.trail.push(r);
  if (state.trail.length % 3 === 0) stage.setTrail(state.trail.slice(-900), colorOf(seg.target));
  if (state.time >= end) {
    state.current = null;
    if (!state.queue.length && (state.generating || state.nextTarget < state.targets.length || $('loop').checked)) state.late++;
    const rest = state.time - end;
    if (state.queue.length) {
      state.current = state.queue.shift();
      state.time = 1 / SAMPLE_RATE + rest;
      pump(state.run);
    }
  }
}

const scratch = { a: null };
function blendFrames(poses, f0, f1, b, J) {
  const out = scratch.a && scratch.a.length === J * TF ? scratch.a : (scratch.a = new Float64Array(J * TF));
  if (b <= 0 || f0 === f1) {
    out.set(poses.subarray(f0 * J * TF, (f0 + 1) * J * TF));
    return out;
  }
  // Linear blend is enough between two 30 Hz frames for display; quaternions renormalised.
  for (let j = 0; j < J; j++) {
    const o0 = (f0 * J + j) * TF;
    const o1 = (f1 * J + j) * TF;
    const o = j * TF;
    for (let k = 0; k < 3; k++) out[o + k] = poses[o0 + k] + (poses[o1 + k] - poses[o0 + k]) * b;
    const s = poses[o0 + 3] * poses[o1 + 3] + poses[o0 + 4] * poses[o1 + 4] + poses[o0 + 5] * poses[o1 + 5] + poses[o0 + 6] * poses[o1 + 6] < 0 ? -1 : 1;
    let n = 0;
    for (let k = 3; k < 7; k++) {
      out[o + k] = poses[o0 + k] + (s * poses[o1 + k] - poses[o0 + k]) * b;
      n += out[o + k] ** 2;
    }
    n = Math.sqrt(n);
    for (let k = 3; k < 7; k++) out[o + k] /= n;
  }
  return out;
}

// ---- mediathèque -------------------------------------------------------------

let catalog = null;
async function loadCatalog() {
  if (catalog) return catalog;
  const res = await fetch(MEDIATHEQUE + 'catalog.json');
  if (!res.ok) throw new Error(`Mediathèque catalog: HTTP ${res.status}`);
  catalog = (await res.json()).filter((item) => item.bvh);
  return catalog;
}

const shortTitle = (item) => item.title.replace(/\s*\(CMU Mocap [^)]*\)\s*$/, '');

async function showCatalog() {
  const list = $('catalog');
  if (list.childElementCount) return;
  try {
    for (const item of await loadCatalog()) {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.title = item.summary || item.title;
      b.setAttribute('aria-pressed', 'false');
      b.innerHTML = `<img loading="lazy" alt="" src="${MEDIATHEQUE + item.thumb}"><span></span>`;
      b.querySelector('span').textContent = shortTitle(item);
      b.addEventListener('click', async () => {
        b.disabled = true;
        await loadClipUrls([MEDIATHEQUE + item.bvh], [shortTitle(item)]);
        b.setAttribute('aria-pressed', 'true');
      });
      li.append(b);
      list.append(li);
    }
  } catch (err) {
    list.textContent = err.message;
  }
}

async function loadDance() {
  const items = await loadCatalog();
  const chosen = DANCE.map((id) => items.find((i) => i.id === id)).filter(Boolean);
  await loadClipUrls(chosen.map((i) => MEDIATHEQUE + i.bvh), chosen.map(shortTitle));
}

async function quickStart() {
  $('quick-start').disabled = true;
  try {
    if (!state.clips.length) {
      try {
        await loadDance();
      } catch (err) {
        console.warn(err);
        await loadClipUrls(SAMPLES.map((p) => SAMPLES_BASE + p));
      }
    }
    if (!state.engine) await loadModel(MODEL_URLS.fp16);
    if (!state.engine || !state.clips.length) return;
    randomTargets();
    $('loop').checked = true;
    play();
  } finally {
    $('quick-start').disabled = false;
  }
}

// ---- UI --------------------------------------------------------------------

function bindUI() {
  $('load-model').addEventListener('click', () => loadModel(MODEL_URLS.fp16));
  $('model-file').addEventListener('change', (e) => e.target.files[0] && loadModel(e.target.files[0]));
  $('clip-files').addEventListener('change', async (e) => {
    for (const file of e.target.files) {
      try {
        addClip(file.name.replace(/\.bvh$/i, ''), await file.text());
      } catch (err) {
        $('model-status').textContent = `${file.name}: ${err.message}`;
      }
    }
    e.target.value = '';
  });
  $('load-samples').addEventListener('click', async () => {
    $('load-samples').disabled = true;
    await loadClipUrls(SAMPLES.map((p) => SAMPLES_BASE + p));
  });
  $('add-target').addEventListener('click', () => {
    const lastPose = state.poses[state.poses.length - 1];
    state.poses.push(newPose(lastPose.clipIndex, lastPose.time, lastPose.x + 1, lastPose.z, lastPose.yaw));
    state.selected = state.poses.length - 1;
    renderPoses();
  });
  $('random-targets').addEventListener('click', () => randomTargets());
  $('quick-start').addEventListener('click', quickStart);
  $('mediatheque').addEventListener('toggle', () => $('mediatheque').open && showCatalog());
  $('play').addEventListener('click', play);
  $('stop').addEventListener('click', stop);
  $('ghosts').addEventListener('change', updateGhosts);
}

function updateButtons() {
  const ready = !!state.engine && state.clips.length > 0 && state.poses.length > 1;
  $('play').disabled = !ready;
  $('play').textContent = state.playing ? 'Restart' : 'Play';
  $('stop').disabled = !state.playing;
  $('add-target').disabled = !state.clips.length;
  $('random-targets').disabled = !state.clips.length;
}

function renderStats() {
  const rows = [
    ['Backend', state.backend || '–'],
    ['Generations', String(state.generations)],
    ['Last generation', state.lastMs ? `${state.lastMs.toFixed(0)} ms` : '–'],
    ['Late handovers', String(state.late)],
    ['Playing', state.playing ? `target ${state.current ? state.current.target : '…'}` : 'no'],
  ];
  $('stats').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
function round(v) {
  return Math.round(v * 4) / 4;
}
function range(a, b) {
  const r = [];
  for (let i = a; i <= b; i++) r.push(i);
  return r;
}

boot().catch((err) => {
  console.error(err);
  $('model-status').textContent = `Could not start: ${err.message || err}`;
});
