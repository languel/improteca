// The improvising dancer: a prompt becomes intentions; the improviser picks
// and transforms learned movements; MotionBricks moves the body between
// their key poses. Generation runs one segment ahead of playback.
//
// URL options: ?model=<onnx url> &backend=wasm &prompt=<text> &seed=<n> &autoplay &clips=<id,id>

import { Stage } from './stage.js';
import { Viz, PRESETS, PARTS, DEFAULT_VIEW } from './viz.js';
import { FIGURES } from './figures.js';
import { Constraints, DEFAULTS as BODY_DEFAULTS } from '../src/engine/constraints.js';
import { startEngine, savedChoice, saveChoice } from './engine-client.js';
import { Skeleton } from '../src/engine/skeleton.js';
import { parseBVH } from '../src/engine/bvh.js';
import { bvhToSoma } from '../src/engine/retarget.js';
import { SomaEngine, SAMPLE_RATE } from '../src/engine/soma.js';
import { Inbetweener } from '../src/engine/inbetweener.js';
import { TF } from '../src/engine/math.js';
import { learn } from '../src/dancer/repertoire.js';
import { Improviser, rng } from '../src/dancer/improviser.js';
import { parsePrompt, describeIntentions } from '../src/dancer/intentions.js';
import { intentionsFromBrain } from '../src/dancer/brains.js';
import { setupBrainPanel, brainSettings, brainName, brainActive } from './brain-panel.js';
import { learnFromVideo } from './video.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);
const MEDIATHEQUE = 'https://languel.github.io/mediatheque/';
// Taught by default: dance and everyday movement that needs no props.
const DEFAULT_CLIPS = [
  'cmu-everyday-waiting-idle-139-02', 'cmu-dance-arching-arms-49-12', 'cmu-ballet-pirouette-05-02',
  'cmu-dance-balance-tilt-49-22', 'cmu-ballet-arabesque-05-03', 'cmu-dance-side-arabesque-49-09',
  'cmu-ballet-pointe-tendue-05-12', 'cmu-ballet-retire-balance-05-14', 'cmu-dance-floor-curl-49-10',
  'cmu-ballet-glissade-05-09', 'cmu-ballet-rond-de-jambe-05-08', 'cmu-ballet-grand-jete-05-17',
  'cmu-ballet-attitude-cambre-05-18', 'cmu-ballet-jete-attitude-05-07', 'cmu-ballet-en-dehors-05-11',
  'cmu-ballet-coupe-jete-05-16', 'cmu-ballet-jete-coupe-05-05', 'cmu-everyday-reach-tiptoe-13-10',
  'cmu-everyday-direct-traffic-13-26', 'cmu-everyday-natural-walk-02-01',
];
const START_AFTER = 6; // clips learned before the dancer starts
const EXAMPLES = [
  'slow and low, like sinking into water',
  'vigorous ballet with jumps and turns',
  'small gestures led by the right hand',
  'travel across the space, unpredictable',
  'calm and still, arms high',
];

const state = {
  skeleton: null,
  modelJointNames: null,
  engine: null,
  backend: '',
  improviser: null,
  sources: [],
  movements: 0,
  loading: null,
  running: false,
  paused: false,
  run: 0,
  player: null,
  pending: [], // targets planned by the improviser, not generated yet
  queue: [], // generated segments waiting to play
  current: null,
  time: 0,
  generating: false,
  late: 0,
  history: [], // prompts and their intentions
  trail: [],
};

const stage = new Stage($('viewport'));
const performerColor = () => getComputedStyle(document.documentElement).getPropertyValue('--performer').trim() || '#2f2c28';
let viz = null;
const VIEW_STORE = 'improteca.view';
const BODY_STORE = 'improteca.body';
const load = (key, fallback) => {
  try {
    return { ...fallback, ...JSON.parse(localStorage.getItem(key) || '{}') };
  } catch {
    return { ...fallback };
  }
};
const store = (key, value) => {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore
  }
};

// ---- boot ----------------------------------------------------------------------

async function boot() {
  const data = await (await fetch('assets/soma/skeleton.json')).json();
  state.skeleton = Skeleton.fromJSON(data);
  state.modelJointNames = data.model_joint_names;
  viz = new Viz(stage, state.skeleton);
  viz.setColor(performerColor());
  await viz.set(params.get('view') && PRESETS[params.get('view')] ? PRESETS[params.get('view')].view : load(VIEW_STORE, DEFAULT_VIEW));
  viz.update(state.skeleton.reference);
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => viz.setColor(performerColor()));
  state.improviser = new Improviser({ skeleton: state.skeleton, movements: [], seed: Number($('seed').value) });
  bindUI();
  // For inspection from the console and automated checks.
  window.improteca = { state, ready: false, pose: (local) => viz.update(local), viz };
  requestAnimationFrame(frame);
  if (params.get('prompt')) $('prompt').value = params.get('prompt');
  if (params.get('seed')) $('seed').value = params.get('seed');
  if (params.has('autoplay')) dance();
}

// ---- loading -------------------------------------------------------------------------

function setStatus(text) {
  $('status').textContent = text;
}

function setProgress(f) {
  const bar = $('progress');
  bar.parentElement.hidden = f === null;
  if (f !== null) bar.style.width = `${(f * 100).toFixed(1)}%`;
}

let catalog = null;
async function loadCatalog() {
  if (catalog) return catalog;
  const res = await fetch(MEDIATHEQUE + 'catalog.json');
  if (!res.ok) throw new Error(`mediathèque catalog: HTTP ${res.status}`);
  catalog = (await res.json()).filter((i) => i.bvh);
  return catalog;
}

const shortTitle = (item) => item.title.replace(/\s*\(CMU Mocap [^)]*\)\s*$/, '');
const tagsOf = (id) => {
  const kind = id.split('-')[1];
  return ['ballet', 'dance', 'everyday'].includes(kind) ? [kind] : [];
};

/** Teach the dancer one clip (BVH text). */
function teach({ id, name, text = null, clip = null, credit = '', tags = null }) {
  if (state.sources.some((s) => s.id === id)) return 0;
  clip ??= bvhToSoma(text, state.skeleton, parseBVH).clip;
  const source = { id, name, clip, tags: tags ?? tagsOf(id), credit };
  const moves = learn(source, state.skeleton);
  state.sources.push(source);
  state.improviser.addMovements(moves);
  state.movements += moves.length;
  renderRepertoire();
  return moves.length;
}

async function teachFromMediatheque(item) {
  const res = await fetch(MEDIATHEQUE + item.bvh);
  if (!res.ok) throw new Error(`${item.id}: HTTP ${res.status}`);
  teach({ id: item.id, name: shortTitle(item), text: await res.text(), credit: item.rights?.credit || item.credit || '' });
}

/** Start (or restart) the model worker with the chosen model and backend. */
async function loadModel() {
  const choice = { model: $('model').value, backend: $('backend').value };
  const client = await startEngine({
    ...choice,
    onProgress: (f, note) => {
      setProgress(f >= 1 ? null : f);
      setStatus(note === 'cached' ? 'Loading the motion model from cache…' : /MB/.test(note) ? `Downloading the motion model… ${note}` : note);
    },
  });
  setProgress(null);
  state.client?.terminate();
  state.client = client;
  if (state.engine) state.engine.run = client.run; // swap under a running dance
  else state.engine = new SomaEngine({ skeleton: state.skeleton, modelJointNames: state.modelJointNames, run: client.run });
  state.backend = client.backend;
  if (client.note) setStatus(client.note);
  renderEngine();
}

function renderEngine() {
  const c = state.client;
  $('engine-stats').textContent = c
    ? `${c.label} · loaded in ${(c.loadMs / 1000).toFixed(1)} s${c.medianMs() ? ` · ${c.medianMs().toFixed(0)} ms per generation (median)` : ''}${state.late ? ` · ${state.late} late handovers` : ''}`
    : 'Not loaded.';
}

function loadEverything() {
  if (state.loading) return state.loading;
  state.loading = (async () => {
    const modelPromise = loadModel();
    // Learn clips while the model downloads.
    const items = await loadCatalog();
    const wanted = (params.get('clips')?.split(',') || DEFAULT_CLIPS).map((id) => items.find((i) => i.id === id)).filter(Boolean);
    let readyForStart;
    const enough = new Promise((r) => (readyForStart = r));
    (async () => {
      for (const [n, item] of wanted.entries()) {
        try {
          await teachFromMediatheque(item);
        } catch (err) {
          console.warn(err);
        }
        if (n + 1 >= Math.min(START_AFTER, wanted.length)) readyForStart();
        // Yield to the page between clips.
        await new Promise((r) => setTimeout(r, 0));
      }
      readyForStart();
    })();
    await Promise.all([modelPromise, enough]);
    setStatus(`Ready: ${state.client.label}.`);
  })();
  state.loading.catch((err) => {
    console.error(err);
    setStatus(`Could not load: ${err.message || err}`);
    state.loading = null;
  });
  return state.loading;
}

// ---- prompt -> intentions --------------------------------------------------------------

async function readPrompt(prompt) {
  if (brainActive()) {
    const name = brainName();
    setStatus(`Asking ${name} how to read the prompt…`);
    try {
      const { intentions, summary } = await intentionsFromBrain(prompt, brainSettings(), {
        onProgress: ({ state, fraction, message }) => {
          if (state === 'loading') setStatus(`${message}${fraction > 0 ? ` ${Math.round(fraction * 100)}%` : ''}`);
        },
      });
      return { intentions, note: `${name}: ${summary || 'read the prompt'}` };
    } catch (err) {
      console.warn(err);
      const { intentions, understood } = parsePrompt(prompt);
      return { intentions, note: `${name} unavailable (${err.message}); read with the built-in vocabulary: ${understood.join('; ') || 'nothing recognised'}.` };
    }
  }
  const { intentions, understood } = parsePrompt(prompt);
  return { intentions, note: understood.length ? `Read as: ${understood.join('; ')}.` : 'No movement words recognised: improvising freely.' };
}

function showIntentions(intentions, note) {
  $('understood').hidden = false;
  const words = describeIntentions(intentions);
  $('intent-chips').replaceChildren(...(words.length ? words : ['free improvisation']).map((w) => {
    const s = document.createElement('span');
    s.className = 'chip on';
    s.textContent = w;
    return s;
  }));
  $('intent-note').textContent = note;
}

async function dance() {
  const prompt = $('prompt').value.trim();
  $('dance').disabled = true;
  try {
    const { intentions, note } = await readPrompt(prompt);
    showIntentions(intentions, note);
    state.history.push({ at: new Date().toISOString(), prompt, intentions });
    state.improviser.setIntentions(intentions);
    await loadEverything();
    if (!state.running) start();
    else {
      // Respond within one movement: drop planned targets not yet generated.
      state.pending = [];
    }
    setStatus(`Dancing · ${state.client.label} · ${state.movements} movements learned.`);
  } catch (err) {
    console.error(err);
    setStatus(`Could not start: ${err.message || err}`);
  } finally {
    $('dance').disabled = false;
  }
}

// ---- performance loop -------------------------------------------------------------------

function start() {
  state.run++;
  state.running = true;
  state.paused = false;
  state.constraints = new Constraints(state.skeleton, bodySettings());
  state.player = new Inbetweener(state.engine, { constraints: state.constraints });
  viz.clear();
  const { targets, entry } = state.improviser.next(null);
  state.player.reset(targets[0].poses);
  state.pending = targets.slice(1).map((t, i) => ({ ...t, entry: i === 0 ? entry : null }));
  state.queue = [];
  state.current = null;
  state.trail = [];
  pump(state.run);
  updateButtons();
  window.improteca.ready = true;
}

/** Keep two segments generated ahead of playback. */
async function pump(run) {
  if (state.generating) return;
  while (run === state.run && state.running && state.queue.length < 2) {
    if (!state.pending.length) {
      const last = state.player.segment
        ? state.player.segment.poses.subarray(state.player.endFrame * state.skeleton.numJoints * TF, (state.player.endFrame + 1) * state.skeleton.numJoints * TF)
        : null;
      const { targets, entry } = state.improviser.next(last ? Float64Array.from(last) : null);
      state.pending = targets.map((t, i) => ({ ...t, entry: i === 0 ? entry : null }));
    }
    const target = state.pending.shift();
    state.generating = true;
    let out;
    try {
      out = await state.player.generate(target);
    } catch (err) {
      console.error(err);
      setStatus(`Generation failed: ${err.message || err}`);
      state.generating = false;
      return;
    } finally {
      state.generating = false;
    }
    if (run !== state.run) return;
    state.player.commit();
    state.queue.push({ poses: out.poses, activity: out.activity, endFrame: state.player.endFrame, entry: target.entry, ms: out.inferenceMs });
  }
}

let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (state.running && !state.paused) advance(dt);
  if (state.running && $('follow').checked && viz.figure) stage.follow(viz.figure.root());
  viz.render(dt);
  stage.render();
  requestAnimationFrame(frame);
}

function advance(dt) {
  const J = state.skeleton.numJoints;
  if (!state.current) {
    if (!state.queue.length) return;
    begin(state.queue.shift(), 0);
  }
  const seg = state.current;
  const end = seg.endFrame / SAMPLE_RATE;
  state.time = Math.min(state.time + dt, end);
  const f = state.time * SAMPLE_RATE;
  const f0 = Math.floor(f);
  const pose = blendFrames(seg.poses, f0, Math.min(f0 + 1, seg.endFrame), f - f0, J);
  // Trails and echoes advance once per model frame (30 Hz); the figure every render frame.
  if (f0 !== state.lastFrame || seg !== state.lastSeg) {
    viz.update(pose, { activity: seg.activity?.[f0] });
    state.lastFrame = f0;
    state.lastSeg = seg;
    state.trail.push(1);
  } else viz.figure?.setPose(pose);
  if (state.time >= end) {
    state.current = null;
    if (state.queue.length) begin(state.queue.shift(), state.time - end);
    else state.late++;
  }
}

function begin(seg, carry) {
  state.current = seg;
  // Frame 0 repeats the last played pose, except for the very first segment.
  state.time = (state.trail.length ? 1 / SAMPLE_RATE : 0) + carry;
  if (seg.entry) showNow(seg.entry);
  pump(state.run);
}

const scratch = { a: null };
function blendFrames(poses, f0, f1, b, J) {
  const out = scratch.a && scratch.a.length === J * TF ? scratch.a : (scratch.a = new Float64Array(J * TF));
  if (b <= 0 || f0 === f1) {
    out.set(poses.subarray(f0 * J * TF, (f0 + 1) * J * TF));
    return out;
  }
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

// ---- display ----------------------------------------------------------------------------

function fillEntry(el, e) {
  el.innerHTML = '<span class="move"></span> <span class="why"></span>';
  el.querySelector('.move').textContent = e.name;
  el.querySelector('.why').textContent = `· ${e.transforms.length ? e.transforms.join(', ') : 'as learned'} · ${e.reasons.join(', ')}`;
}

function renderPhrases(current) {
  const mem = state.improviser.memory();
  const form = $('form');
  form.replaceChildren();
  mem.form.forEach((label, i) => {
    if (i) form.append(' · ');
    const span = document.createElement('span');
    span.textContent = label;
    if (i === mem.form.length - 1 && current) span.className = 'current';
    form.append(span);
  });
  if (!mem.form.length) form.textContent = '—';
  $('motifs').replaceChildren(...mem.motifs.map((mo) => {
    const li = document.createElement('li');
    li.textContent = `${mo.name}: ${mo.items.join(' + ') || '…'}`;
    li.title = li.textContent;
    return li;
  }));
}

function showNow(e) {
  const now = $('now');
  now.replaceChildren();
  const p = document.createElement('p');
  fillEntry(p, e);
  if (e.phrase) {
    const tag = document.createElement('span');
    tag.className = 'phrase';
    tag.textContent = e.phrase;
    p.prepend(tag);
  }
  now.append(p);
  renderPhrases(true);
  const li = document.createElement('li');
  fillEntry(li, e);
  $('log').prepend(li);
  while ($('log').childElementCount > 200) $('log').lastElementChild.remove();
}

function renderRepertoire() {
  $('repertoire').textContent = `${state.movements} movements learned from ${state.sources.length} clips: ${state.sources.map((s) => s.name).join(', ')}.`;
  for (const b of document.querySelectorAll('#catalog button')) {
    if (state.sources.some((s) => s.id === b.dataset.id)) b.setAttribute('aria-pressed', 'true');
  }
  for (const b of document.querySelectorAll('#films button')) {
    if (state.sources.some((s) => s.id === b.dataset.id)) b.setAttribute('aria-pressed', 'true');
  }
}

async function showCatalog() {
  const list = $('catalog');
  if (list.childElementCount) return;
  try {
    for (const item of await loadCatalog()) {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.dataset.id = item.id;
      b.title = item.summary || item.title;
      b.setAttribute('aria-pressed', String(state.sources.some((s) => s.id === item.id)));
      b.innerHTML = `<img loading="lazy" alt="" src="${MEDIATHEQUE + item.thumb}"><span></span>`;
      b.querySelector('span').textContent = shortTitle(item);
      b.addEventListener('click', async () => {
        b.disabled = true;
        try {
          await teachFromMediatheque(item);
        } catch (err) {
          setStatus(err.message);
        }
        b.disabled = false;
      });
      li.append(b);
      list.append(li);
    }
  } catch (err) {
    list.textContent = err.message;
  }
}

const films = (items) => items.filter((i) => ['video', 'animation'].includes(i.kind) && i.proxy && (i.demos || []).includes('pose') && (i.tags || []).includes('fullbody'));

let filmCatalog = null;
async function loadFilms() {
  if (filmCatalog) return filmCatalog;
  const res = await fetch(MEDIATHEQUE + 'catalog.json');
  if (!res.ok) throw new Error(`mediathèque catalog: HTTP ${res.status}`);
  filmCatalog = films(await res.json());
  return filmCatalog;
}

async function learnFilm({ id, name, url, credit = '' }) {
  setStatus(`Watching ${name}…`);
  setProgress(0);
  try {
    const { clip, frames, tracked } = await learnFromVideo(url, state.skeleton, {
      onProgress: (f, note) => {
        setProgress(f);
        setStatus(`${name}: ${note}`);
      },
    });
    const n = teach({ id, name, clip, credit, tags: ['film'] });
    setStatus(`Learned ${n} movements from ${name} (dancer found in ${tracked} of ${frames} frames).`);
    return n;
  } catch (err) {
    console.error(err);
    setStatus(`${name}: ${err.message || err}`);
    return 0;
  } finally {
    setProgress(null);
  }
}

async function showFilms() {
  const list = $('films');
  if (list.childElementCount) return;
  try {
    for (const item of await loadFilms()) {
      const li = document.createElement('li');
      const b = document.createElement('button');
      b.dataset.id = item.id;
      b.title = item.summary || item.title;
      b.setAttribute('aria-pressed', String(state.sources.some((s) => s.id === item.id)));
      b.innerHTML = `<img loading="lazy" alt="" src="${MEDIATHEQUE + item.thumb}"><span></span>`;
      b.querySelector('span').textContent = item.title.replace(/\s*\([^)]*\)\s*$/, '');
      b.addEventListener('click', async () => {
        b.disabled = true;
        const n = await learnFilm({ id: item.id, name: b.querySelector('span').textContent, url: MEDIATHEQUE + item.proxy, credit: item.rights?.credit || '' });
        if (n) b.setAttribute('aria-pressed', 'true');
        b.disabled = false;
      });
      li.append(b);
      list.append(li);
    }
  } catch (err) {
    list.textContent = err.message;
  }
}

function exportLog() {
  const data = {
    seed: state.improviser.seed,
    prompts: state.history,
    repertoire: state.sources.map((s) => ({ id: s.id, name: s.name, credit: s.credit })),
    choices: state.improviser.log,
    phrases: state.improviser.memory(),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `improteca-performance-${Date.now()}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---- body (constraints) and view ------------------------------------------------------

const BODY_KEYS = ['limits', 'feet', 'floor', 'balance', 'smooth'];
function bodySettings() {
  return Object.fromEntries(BODY_KEYS.map((k) => [k, Number($(`body-${k}`).value)]));
}

function bindBody() {
  const saved = load(BODY_STORE, BODY_DEFAULTS);
  for (const k of BODY_KEYS) {
    const input = $(`body-${k}`);
    input.value = saved[k];
    const out = input.parentElement.querySelector('output');
    const show = () => (out.textContent = Number(input.value) === 0 ? 'off' : `${Math.round(input.value * 100)}%`);
    show();
    input.addEventListener('input', () => {
      show();
      const settings = bodySettings();
      state.constraints?.set(settings);
      store(BODY_STORE, settings);
    });
  }
}

function bindView() {
  const v = viz.view;
  const fig = $('view-figure');
  fig.replaceChildren(...Object.entries(FIGURES).map(([k, d]) => new Option(d.label, k)));
  fig.value = v.figure;
  const parts = $('view-parts');
  parts.replaceChildren(...Object.entries(PARTS).map(([k, p]) => {
    const label = document.createElement('label');
    label.className = 'check';
    label.innerHTML = `<input type="checkbox" value="${k}"> <span class="swatch" style="background:${p.color}"></span>${p.label}`;
    label.querySelector('input').checked = v.trails.includes(k);
    return label;
  }));
  $('view-presets').replaceChildren(...Object.entries(PRESETS).map(([k, p]) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.type = 'button';
    b.textContent = p.label;
    b.addEventListener('click', async () => {
      await viz.set(p.view);
      syncView();
      store(VIEW_STORE, viz.view);
    });
    return b;
  }));
  const syncView = () => {
    const w = viz.view;
    fig.value = w.figure;
    for (const c of parts.querySelectorAll('input')) c.checked = w.trails.includes(c.value);
    $('view-trail').value = w.trailSeconds;
    $('view-ribbons').checked = w.ribbons;
    $('view-echo').value = w.echoEvery;
    $('view-echo-count').value = w.echoCount;
    $('view-footprints').checked = w.footprints;
    $('view-activity').checked = w.activity;
    $('view-bg').value = w.background;
    for (const id of ['view-trail', 'view-echo', 'view-echo-count']) $(id).dispatchEvent(new Event('show'));
  };
  const outputs = { 'view-trail': (x) => `${x} s`, 'view-echo': (x) => (Number(x) ? `every ${x} s` : 'off'), 'view-echo-count': (x) => `${x}` };
  for (const [id, f] of Object.entries(outputs)) {
    const show = () => ($(id).parentElement.querySelector('output').textContent = f($(id).value));
    $(id).addEventListener('input', show);
    $(id).addEventListener('show', show);
  }
  const apply = async () => {
    await viz.set({
      figure: fig.value,
      trails: [...parts.querySelectorAll('input:checked')].map((c) => c.value),
      trailSeconds: Number($('view-trail').value),
      ribbons: $('view-ribbons').checked,
      echoEvery: Number($('view-echo').value),
      echoCount: Number($('view-echo-count').value),
      footprints: $('view-footprints').checked,
      activity: $('view-activity').checked,
      background: $('view-bg').value,
    });
    store(VIEW_STORE, viz.view);
  };
  for (const el of document.querySelectorAll('#view-panel input, #view-panel select')) el.addEventListener('change', apply);
  $('view-clear').addEventListener('click', () => viz.clear());
  syncView();
}

// ---- UI -------------------------------------------------------------------------------

function bindUI() {
  for (const text of EXAMPLES) {
    const b = document.createElement('button');
    b.className = 'chip';
    b.type = 'button';
    b.textContent = text;
    b.addEventListener('click', () => {
      $('prompt').value = text;
      dance();
    });
    $('examples').append(b);
  }
  $('dance').addEventListener('click', dance);
  bindBody();
  bindView();
  const choice = savedChoice(params);
  for (const [id, value] of [['model', choice.model], ['backend', choice.backend]]) {
    if (![...$(id).options].some((o) => o.value === value)) $(id).add(new Option(value, value));
    $(id).value = value;
  }
  $('apply-engine').addEventListener('click', async () => {
    saveChoice({ model: $('model').value, backend: $('backend').value });
    if (!state.client) return setStatus('Saved: used when the dancer starts.');
    $('apply-engine').disabled = true;
    try {
      await loadModel();
      setStatus(`Switched to ${state.client.label}.`);
    } catch (err) {
      setStatus(`Could not switch: ${err.message || err}`);
    } finally {
      $('apply-engine').disabled = false;
    }
  });
  setInterval(renderEngine, 1000);
  $('prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) dance();
  });
  $('pause').addEventListener('click', () => {
    state.paused = !state.paused;
    updateButtons();
  });
  const setSeed = (n) => {
    $('seed').value = n;
    state.improviser.seed = n;
    state.improviser.random = rng(n);
  };
  $('seed').addEventListener('change', () => setSeed(Number($('seed').value) || 1));
  $('reseed').addEventListener('click', () => setSeed(1 + Math.floor(Math.random() * 99999)));
  $('export').addEventListener('click', exportLog);
  $('repertoire-details').addEventListener('toggle', () => {
    if (!$('repertoire-details').open) return;
    showCatalog();
    showFilms();
  });
  $('video-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    await learnFilm({ id: `file:${file.name}`, name: file.name.replace(/\.[^.]+$/, ''), url });
    URL.revokeObjectURL(url);
    e.target.value = '';
  });
  $('bvh').addEventListener('change', async (e) => {
    for (const file of e.target.files) {
      try {
        const n = teach({ id: `file:${file.name}`, name: file.name.replace(/\.bvh$/i, ''), text: await file.text() });
        setStatus(`Learned ${n} movements from ${file.name}.`);
      } catch (err) {
        setStatus(`${file.name}: ${err.message}`);
      }
    }
    e.target.value = '';
  });
  setupBrainPanel();
}

function updateButtons() {
  $('pause').disabled = !state.running;
  $('pause').textContent = state.paused ? 'Resume' : 'Pause';
}

boot().catch((err) => {
  console.error(err);
  setStatus(`Could not start: ${err.message || err}`);
});
