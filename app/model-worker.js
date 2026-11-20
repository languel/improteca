// Runs the MotionBricks ONNX model in a worker, so the page never stalls.
// Loads either a single file (NVIDIA's FP16) or gzip parts listed in a
// manifest (the 8-bit build), caches what it downloads, and runs inference
// on WebGPU or WebAssembly.
//
// Messages in:  { type: 'init', source, backend }   source: { kind: 'file', url } | { kind: 'parts', manifestUrl }
//               { type: 'run', id, inputs }
// Messages out: { type: 'progress', f, note } | { type: 'ready', backend, ms, bytes } | { type: 'error', id?, text }
//               { type: 'result', id, pred, tokens, ms }

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

const CACHE = 'improteca-models-v2';
const SHAPES = {
  global_root_values: ['float32', [1, 8, 5]],
  local_root_values: ['float32', [1, 8, 4]],
  local_poses: ['float32', [1, 8, 267]],
  has_global_root_values: ['int32', [1, 8]],
  has_local_root_values: ['int32', [1, 8]],
  has_local_poses: ['int32', [1, 8]],
  allowed_pred_num_tokens: ['int32', [1, 11]],
};
let session = null;

const progress = (f, note) => postMessage({ type: 'progress', f, note });

async function openCache() {
  try {
    return await caches.open(CACHE);
  } catch {
    return null; // private mode, blocked storage: no caching
  }
}

/** A response from the cache, else the network (then cached). Cache key may differ from the URL. */
async function cachedFetch(cache, url, key = url) {
  if (cache) {
    const hit = await cache.match(key);
    if (hit) return { res: hit, cached: true };
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  if (cache) {
    try {
      await cache.put(key, res.clone());
    } catch {
      // storage full: carry on uncached
    }
  }
  return { res, cached: false };
}

async function readAll(stream, total, onBytes) {
  const out = new Uint8Array(total);
  let off = 0;
  const reader = stream.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (off + value.byteLength > out.length) throw new Error('model larger than expected');
    out.set(value, off);
    off += value.byteLength;
    onBytes(off);
  }
  if (off !== total) throw new Error(`model is ${off} bytes, expected ${total}`);
  return out;
}

async function loadFile(url) {
  const cache = await openCache();
  const { res, cached } = await cachedFetch(cache, url);
  const total = Number(res.headers.get('content-length')) || 0;
  if (!total) return new Uint8Array(await res.arrayBuffer());
  return readAll(res.body, total, (n) => progress(n / total, cached ? 'cached' : `${(n / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`));
}

async function loadParts(manifestUrl) {
  const manifest = await (await fetch(manifestUrl, { cache: 'no-cache' })).json();
  const cache = await openCache();
  const base = new URL('.', manifestUrl);
  let got = 0;
  let i = 0;
  let reader = null;
  let anyNetwork = false;
  // Stream the parts in order through the browser's gzip decoder: never two copies in memory.
  const packed = new ReadableStream({
    async pull(ctrl) {
      for (;;) {
        if (!reader) {
          if (i >= manifest.parts.length) {
            ctrl.close();
            return;
          }
          const url = new URL(manifest.parts[i++], base).href;
          const { res, cached } = await cachedFetch(cache, url, `${url}?sha=${manifest.sha256}`);
          anyNetwork ||= !cached;
          reader = res.body.getReader();
        }
        const { done, value } = await reader.read();
        if (done) {
          reader = null;
          continue;
        }
        got += value.byteLength;
        progress(got / manifest.packed, anyNetwork ? `${(got / 1e6).toFixed(0)} of ${(manifest.packed / 1e6).toFixed(0)} MB` : 'cached');
        ctrl.enqueue(value);
        return;
      }
    },
  });
  return readAll(packed.pipeThrough(new DecompressionStream('gzip')), manifest.bytes, () => {});
}

onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const bytes = m.source.kind === 'parts' ? await loadParts(m.source.manifestUrl) : await loadFile(m.source.url);
      progress(1, 'Starting the model…');
      const order = m.backend === 'wasm' || !self.navigator.gpu ? ['wasm'] : ['webgpu', 'wasm'];
      let lastError = null;
      for (const backend of order) {
        try {
          const t0 = performance.now();
          session = await ort.InferenceSession.create(bytes, { executionProviders: [backend], graphOptimizationLevel: 'all' });
          postMessage({ type: 'ready', backend, ms: performance.now() - t0, bytes: bytes.byteLength, threads: ort.env.wasm.numThreads });
          return;
        } catch (err) {
          lastError = err;
        }
      }
      throw lastError;
    }
    if (m.type === 'run') {
      const feeds = {};
      for (const [name, [type, dims]] of Object.entries(SHAPES)) feeds[name] = new ort.Tensor(type, m.inputs[name], dims);
      const t0 = performance.now();
      const out = await session.run(feeds);
      const ms = performance.now() - t0;
      const pred = new Float32Array(out.pred_global_motions.data);
      postMessage({ type: 'result', id: m.id, pred, tokens: Number(out.pred_num_tokens.data[0]), ms }, [pred.buffer]);
    }
  } catch (err) {
    postMessage({ type: 'error', id: m.id, text: String(err?.message || err) });
  }
};
