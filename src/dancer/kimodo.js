// Kimodo (NVIDIA's text-to-motion model on the SOMA skeleton), through the
// local server in tools/kimodo_server.py. Optional: without it, the director
// borrows poses from learned clips instead.
//
// A generated motion comes back as a Clip on improteca's rig, in its own
// frame (starting at the origin, facing +Z), with foot contacts per frame.

import { Clip } from '../engine/bvh.js';
import { TF } from '../engine/math.js';

export const DEFAULT_URL = 'http://127.0.0.1:5320';

export class KimodoClient {
  /**
   * @param {import('../engine/skeleton.js').Skeleton} skeleton
   * @param {{ url?: string, fetch?: typeof fetch }} [options]
   */
  constructor(skeleton, { url = DEFAULT_URL, fetch: f = globalThis.fetch?.bind(globalThis) } = {}) {
    this.skeleton = skeleton;
    this.url = url.replace(/\/+$/, '');
    this.fetch = f;
    this.ready = false;
    this.info = null;
    this.error = null;
  }

  /** Is the server there? Sets ready, info and error. */
  async check(timeoutMs = 1500) {
    try {
      const ctl = new AbortController();
      const timer = setTimeout(() => ctl.abort(), timeoutMs);
      const res = await this.fetch(`${this.url}/health`, { signal: ctl.signal });
      clearTimeout(timer);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      this.info = await res.json();
      if (this.info.joints !== this.skeleton.numJoints) throw new Error(`the server speaks ${this.info.joints} joints, the page ${this.skeleton.numJoints}`);
      this.ready = true;
      this.error = null;
    } catch (err) {
      this.ready = false;
      this.error = err.name === 'AbortError' ? 'not running' : err.message || String(err);
    }
    return this.ready;
  }

  /**
   * One motion.
   * @param {{ text: string, seconds: number, seed?: number, path?: { frames: number[], xz: number[][], heading?: number[] } }} req
   * @returns {Promise<{ clip: Clip, contacts: number[][], ms: number, text: string }>}
   */
  async generate(req) {
    const res = await this.fetch(`${this.url}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || `HTTP ${res.status}`);
    const bytes = Uint8Array.from(atob(body.local), (c) => c.charCodeAt(0));
    const f32 = new Float32Array(bytes.buffer);
    const J = this.skeleton.numJoints;
    if (f32.length !== body.frames * J * TF) throw new Error(`Kimodo sent ${f32.length} values, expected ${body.frames * J * TF}`);
    return { clip: new Clip(this.skeleton, body.fps, Float64Array.from(f32)), contacts: body.contacts, ms: body.ms, text: req.text };
  }
}
