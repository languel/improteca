// Download (and cache) the MotionBricks ONNX model and open an ONNX Runtime
// Web session: WebGPU when the browser has it, WebAssembly otherwise.

import * as ort from 'onnxruntime-web';

const ORT_DIST = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/';
const CACHE = 'improteca-models-v1';

ort.env.wasm.wasmPaths = ORT_DIST;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;

/** Fetch a URL as bytes, reporting progress, from the Cache API when present. */
export async function fetchModel(url, onProgress = () => {}) {
  let cache = null;
  try {
    cache = await caches.open(CACHE);
    const hit = await cache.match(url);
    if (hit) {
      onProgress(1, 'cached');
      return new Uint8Array(await hit.arrayBuffer());
    }
  } catch {
    cache = null;
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed: HTTP ${res.status}`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(total ? received / total : 0, `${(received / 1e6).toFixed(0)} MB`);
  }
  const bytes = new Uint8Array(received);
  let o = 0;
  for (const c of chunks) {
    bytes.set(c, o);
    o += c.length;
  }
  if (cache) {
    try {
      await cache.put(url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } }));
    } catch {
      // Storage full or blocked: carry on without caching.
    }
  }
  return bytes;
}

/** Open a session, preferring WebGPU. Returns { ort, session, backend }. */
export async function openSession(bytes, prefer = 'webgpu') {
  const order = prefer === 'wasm' || !navigator.gpu ? ['wasm'] : ['webgpu', 'wasm'];
  let lastError = null;
  for (const backend of order) {
    try {
      const session = await ort.InferenceSession.create(bytes, {
        executionProviders: [backend],
        graphOptimizationLevel: 'all',
      });
      return { ort, session, backend };
    } catch (err) {
      lastError = err;
      console.warn(`ONNX Runtime ${backend} failed`, err);
    }
  }
  throw lastError;
}
