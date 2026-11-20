// The page side of the model worker: picks a model and backend, starts the
// worker, and exposes run(inputs) with the same shape as ortRunner(), so
// SomaEngine works unchanged.
//
// Models:   'fp16'  NVIDIA's FP16 file (380 MB), loaded from NVIDIA's repository
//           'int8'  improteca's 8-bit build (210 MB gzipped, served with this site)
//           any URL ending in .onnx, or a manifest.json of parts
// Backends: 'webgpu' (falls back to WebAssembly if it fails), 'wasm'
// 'auto' picks FP16 on WebGPU, 8-bit on WebAssembly.

import { MODEL_URLS } from '../src/engine/ort.js';

export const MODELS = {
  fp16: { label: 'FP16 · NVIDIA · 380 MB', source: { kind: 'file', url: MODEL_URLS.fp16 } },
  int8: { label: '8-bit · smaller, faster on CPU · 210 MB', source: { kind: 'parts', manifestUrl: new URL('../models/int8/manifest.json', import.meta.url).href } },
};

async function hasWebGPU() {
  try {
    return !!(navigator.gpu && (await navigator.gpu.requestAdapter()));
  } catch {
    return false;
  }
}

async function exists(url) {
  try {
    const r = await fetch(url, { method: 'HEAD', cache: 'no-cache' });
    return r.ok;
  } catch {
    return false;
  }
}

/** Resolve 'auto' and URLs into { model, source, backend, note }. */
export async function resolveChoice(model = 'auto', backend = 'auto') {
  const gpu = await hasWebGPU();
  const be = backend === 'auto' ? (gpu ? 'webgpu' : 'wasm') : backend;
  let m = model === 'auto' ? (be === 'webgpu' ? 'fp16' : 'int8') : model;
  let note = '';
  let source;
  if (MODELS[m]) source = MODELS[m].source;
  else if (/manifest\.json$/.test(m)) source = { kind: 'parts', manifestUrl: new URL(m, location.href).href };
  else source = { kind: 'file', url: new URL(m, location.href).href };
  if (source.kind === 'parts' && !(await exists(source.manifestUrl))) {
    note = '8-bit build not found here; using FP16.';
    m = 'fp16';
    source = MODELS.fp16.source;
  }
  return { model: m, source, backend: be, note };
}

/**
 * Start the model in a worker.
 * @returns {Promise<{ run, backend, model, label, loadMs, stats, terminate }>}
 */
export async function startEngine({ model = 'auto', backend = 'auto', onProgress = () => {} } = {}) {
  const choice = await resolveChoice(model, backend);
  const worker = new Worker(new URL('./model-worker.js', import.meta.url), { type: 'module' });
  const pending = new Map();
  let nextId = 1;
  const times = [];
  const ready = new Promise((resolve, reject) => {
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'progress') onProgress(m.f, m.note);
      else if (m.type === 'ready') resolve(m);
      else if (m.type === 'result') {
        times.push(m.ms);
        if (times.length > 30) times.shift();
        pending.get(m.id)?.resolve({ pred_global_motions: m.pred, pred_num_tokens: [m.tokens] });
        pending.delete(m.id);
      } else if (m.type === 'error') {
        if (m.id && pending.has(m.id)) {
          pending.get(m.id).reject(new Error(m.text));
          pending.delete(m.id);
        } else reject(new Error(m.text));
      }
    };
    worker.onerror = (e) => reject(new Error(e.message || 'model worker failed'));
  });
  worker.postMessage({ type: 'init', source: choice.source, backend: choice.backend });
  const info = await ready;
  const run = (inputs) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      // Copy, then transfer the copies: the caller keeps its arrays.
      const copy = {};
      const transfer = [];
      for (const [k, v] of Object.entries(inputs)) {
        copy[k] = v.slice();
        transfer.push(copy[k].buffer);
      }
      worker.postMessage({ type: 'run', id, inputs: copy }, transfer);
    });
  return {
    run,
    model: choice.model,
    backend: info.backend,
    label: `${MODELS[choice.model]?.label.split(' · ')[0] || decodeURIComponent(choice.model.split('/').pop()).replace(/^blob:.*/, 'local file')} on ${info.backend === 'webgpu' ? 'WebGPU' : `WebAssembly${info.threads > 1 ? ` (${info.threads} threads)` : ''}`}`,
    note: choice.note,
    loadMs: info.ms,
    /** Median generation time over the last 30 generations (ms). */
    medianMs: () => {
      if (!times.length) return 0;
      const s = [...times].sort((a, b) => a - b);
      return s[Math.floor(s.length / 2)];
    },
    terminate: () => worker.terminate(),
  };
}

const STORE = 'improteca.engine';
/** Remembered choice (per device), with URL parameters taking precedence. */
export function savedChoice(params) {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem(STORE) || '{}');
  } catch {
    saved = {};
  }
  return { model: params.get('model') || saved.model || 'auto', backend: params.get('backend') || saved.backend || 'auto' };
}
export function saveChoice(choice) {
  try {
    localStorage.setItem(STORE, JSON.stringify(choice));
  } catch {
    // ignore
  }
}
