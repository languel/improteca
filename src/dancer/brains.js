// Prompt -> intentions with a language model ("brain"). Optional: the lexicon
// in intentions.js always works without one. Brains:
//
//   - in the browser, on WebGPU: Liquid AI's LFM3-Omni-2B or LFM3-1.7B
//     (weights download once from Hugging Face; nothing leaves the machine);
//   - on this machine: Ollama, LM Studio or any OpenAI-compatible server;
//   - in the cloud, with your own key: Anthropic Claude, OpenAI, OpenRouter,
//     Google Gemini, GitHub Models, or any OpenAI-compatible API.
//
// Keys are sent only to the provider they belong to. The provider list and
// request shapes follow languel/seloai (src/ai/localApiClient.ts).

import { NEUTRAL } from './improviser.js';

export const ANTHROPIC_SDK_URL = 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@1.6.0/+esm';

/** Brains, in the order the page lists them. kind: none | webgpu | anthropic | gemini | openai (OpenAI-compatible). */
export const PROVIDERS = [
  { id: 'none', label: 'Built-in vocabulary (no model)', kind: 'none' },
  { id: 'lfm3-omni', label: 'LFM3-Omni-2B · WebGPU · in the browser', kind: 'webgpu', model: 'LiquidAI/LFM3-Omni-2B-GGUF', size: 'about 1.6 GB' },
  { id: 'lfm3', label: 'LFM3-1.7B · WebGPU · in the browser', kind: 'webgpu', model: 'LiquidAI/LFM3-1.7B-GGUF', size: 'about 1.1 GB' },
  { id: 'ollama', label: 'Ollama (this machine)', kind: 'openai', baseUrl: 'http://localhost:11434/v1', model: 'lfm3', key: false },
  { id: 'lmstudio', label: 'LM Studio (this machine)', kind: 'openai', baseUrl: 'http://localhost:1234/v1', model: 'local-model', key: false },
  { id: 'anthropic', label: 'Anthropic Claude', kind: 'anthropic', baseUrl: 'https://api.anthropic.com', model: 'claude-opus-5-5', key: true },
  { id: 'openai', label: 'OpenAI', kind: 'openai', baseUrl: 'https://api.openai.com/v1', model: 'gpt-4o-mini', key: true },
  { id: 'openrouter', label: 'OpenRouter', kind: 'openai', baseUrl: 'https://openrouter.ai/api/v1', model: 'google/gemini-2.5-flash', key: true },
  { id: 'gemini', label: 'Google Gemini', kind: 'gemini', baseUrl: 'https://generativelanguage.googleapis.com/v1beta', model: 'gemini-2.5-flash', key: true },
  { id: 'github', label: 'GitHub Models', kind: 'openai', baseUrl: 'https://models.inference.ai.azure.com', model: 'gpt-4o-mini', key: true },
  { id: 'custom', label: 'Other OpenAI-compatible API', kind: 'openai', baseUrl: 'http://localhost:8080/v1', model: 'default', key: false },
];
export const provider = (id) => PROVIDERS.find((p) => p.id === id) ?? PROVIDERS[0];

// ---- what the model is asked for ------------------------------------------------------------

const TAGS = ['ballet', 'dance', 'everyday', 'film'];
const FOCUS = ['handL', 'handR', 'footL', 'footR', 'head', 'arms', 'legs'];
const QUALITIES = {
  energy: 'Calm (0) to vigorous (1).',
  level: 'Low, near the floor (0) to high, lifted (1).',
  size: 'Small and close to the body (0) to large and extended (1).',
  travel: 'On the spot (0) to travelling across the space (1).',
  turn: 'No turning (0) to turning and spinning (1).',
  armsUp: 'Arms low (0) to arms overhead (1).',
  jump: 'Grounded (0) to jumping, airborne (1).',
  stillness: 'Continuous flow (0) to pauses and holds (1).',
};
const quality = (description) => ({ type: ['number', 'null'], description: `${description} 0 to 1, or null for no preference.` });

/** JSON schema for the intentions object (structured outputs: all keys required, no extras). */
export const INTENTIONS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: [...Object.keys(QUALITIES), 'focus', 'tags', 'avoid', 'tempo', 'variety', 'mirror', 'retrograde', 'summary'],
  properties: {
    ...Object.fromEntries(Object.entries(QUALITIES).map(([k, d]) => [k, quality(d)])),
    focus: { type: ['string', 'null'], enum: [...FOCUS, null], description: 'Body part that should lead, or null.' },
    tags: { type: 'array', items: { type: 'string', enum: TAGS }, description: 'Preferred material: ballet, dance (contemporary/modern), everyday (pedestrian gestures), film (movement learned from archival films).' },
    avoid: { type: 'array', items: { type: 'string', enum: TAGS }, description: 'Material to avoid.' },
    tempo: { type: 'number', description: 'Playback tempo: 1 natural, 0.6 much slower, 1.4 much faster.' },
    variety: { type: 'number', description: 'Repetitive (0) to unpredictable (1); 0.5 default.' },
    mirror: { type: 'number', description: 'Weight for mirrored (left/right swapped) versions; 0.35 default.' },
    retrograde: { type: 'number', description: 'Weight for movements danced backwards; 0.1 default.' },
    summary: { type: 'string', description: 'One short sentence: how you read the prompt as movement.' },
  },
};

export const SYSTEM = `You translate a choreographer's prompt into intentions for an improvising virtual dancer.
The dancer knows recorded movements (ballet, contemporary dance and everyday gestures from motion capture)
and chooses among them by these qualities. Read imagery and metaphor as movement qualities
(e.g. "like seaweed" = continuous, low energy, arms leading; "a storm" = vigorous, travelling, turning).
Set a quality only when the prompt implies it; otherwise null. Keep tempo between 0.6 and 1.4.`;

/** For models without structured outputs: the same schema, spelled out. */
const JSON_INSTRUCTIONS = `Answer with one JSON object and nothing else (no prose, no code fence), with exactly these keys:
${Object.entries(INTENTIONS_SCHEMA.properties).map(([k, v]) => `- "${k}": ${v.description}${v.enum ? ` One of: ${v.enum.filter(Boolean).join(', ')}, or null.` : ''}${v.items?.enum ? ` Values from: ${v.items.enum.join(', ')}.` : ''}`).join('\n')}
Example: {"energy":0.2,"level":0.1,"size":null,"travel":null,"turn":null,"armsUp":null,"jump":0,"stillness":0.6,"focus":"arms","tags":[],"avoid":[],"tempo":0.8,"variety":0.5,"mirror":0.35,"retrograde":0.1,"summary":"Slow, low and sustained, arms leading."}`;

export const messagesFor = (prompt, structured) => [
  { role: 'system', content: structured ? SYSTEM : `${SYSTEM}\n\n${JSON_INSTRUCTIONS}` },
  { role: 'user', content: prompt },
];

/** The first JSON object in a model's answer (tolerates thinking, prose and code fences). */
export function extractJSON(text) {
  const clean = String(text).replace(/<think>[\s\S]*?<\/think>/gi, '').replace(/```(?:json)?/gi, '');
  const start = clean.indexOf('{');
  if (start < 0) throw new Error('The model did not answer with JSON.');
  let depth = 0;
  let inString = false;
  for (let i = start; i < clean.length; i++) {
    const c = clean[i];
    if (inString) {
      if (c === '\\') i++;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === '{') depth++;
    else if (c === '}' && --depth === 0) return JSON.parse(clean.slice(start, i + 1));
  }
  throw new Error('The model\'s JSON was cut off.');
}

/** Whatever the model returned, as valid intentions (unknown keys dropped, numbers clamped). */
export function sanitize(raw) {
  const out = { ...NEUTRAL };
  const num = (v, lo, hi) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : null);
  for (const k of Object.keys(QUALITIES)) out[k] = num(raw?.[k], 0, 1);
  out.focus = FOCUS.includes(raw?.focus) ? raw.focus : null;
  const tags = (v) => (Array.isArray(v) ? [...new Set(v.filter((t) => TAGS.includes(t)))] : []);
  out.tags = tags(raw?.tags);
  out.avoid = tags(raw?.avoid);
  const keep = (k, lo, hi) => {
    const v = num(raw?.[k], lo, hi);
    if (v !== null) out[k] = v;
  };
  keep('tempo', 0.6, 1.4);
  keep('variety', 0, 1);
  keep('mirror', 0, 1);
  keep('retrograde', 0, 1);
  return { intentions: out, summary: typeof raw?.summary === 'string' ? raw.summary.slice(0, 240) : '' };
}

// ---- HTTP providers -------------------------------------------------------------------------

const trimUrl = (url) => String(url || '').trim().replace(/\/+$/, '');

function headersFor(p, apiKey) {
  const h = { 'Content-Type': 'application/json' };
  const key = apiKey?.trim();
  if (key && p.kind !== 'gemini') h.Authorization = `Bearer ${key}`;
  if (p.id === 'openrouter') {
    if (globalThis.location) h['HTTP-Referer'] = globalThis.location.origin;
    h['X-Title'] = 'improteca';
  }
  return h;
}

/**
 * The HTTP request for one chat turn (OpenAI-compatible or Gemini). Pure, for tests.
 * json: ask for the intentions schema as structured output where the provider supports it.
 */
export function buildRequest(p, { baseUrl, model, apiKey, messages, json = false, maxTokens = 1024 }) {
  const base = trimUrl(baseUrl || p.baseUrl);
  if (p.kind === 'gemini') {
    const system = messages.filter((m) => m.role === 'system').map((m) => m.content).join('\n\n');
    const key = apiKey?.trim() ? `?key=${encodeURIComponent(apiKey.trim())}` : '';
    return {
      url: `${base}/models/${encodeURIComponent(String(model).replace(/^models\//, ''))}:generateContent${key}`,
      headers: headersFor(p, apiKey),
      body: {
        ...(system ? { systemInstruction: { parts: [{ text: system }] } } : {}),
        contents: messages.filter((m) => m.role !== 'system').map((m) => ({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content }] })),
        generationConfig: { maxOutputTokens: maxTokens, temperature: 0.3, ...(json ? { responseMimeType: 'application/json' } : {}) },
      },
    };
  }
  return {
    url: base.endsWith('/chat/completions') ? base : `${base}/chat/completions`,
    headers: headersFor(p, apiKey),
    body: {
      model,
      messages,
      max_tokens: maxTokens,
      temperature: 0.3,
      ...(json ? { response_format: { type: 'json_schema', json_schema: { name: 'intentions', strict: true, schema: INTENTIONS_SCHEMA } } } : {}),
    },
  };
}

async function postChat(p, settings, messages, json) {
  const spec = buildRequest(p, { ...settings, messages, json });
  const res = await fetch(spec.url, { method: 'POST', headers: spec.headers, body: JSON.stringify(spec.body) });
  if (!res.ok) {
    const err = new Error(`${p.label}: HTTP ${res.status} ${(await res.text().catch(() => '')).slice(0, 160)}`);
    err.status = res.status;
    throw err;
  }
  const data = await res.json();
  if (p.kind === 'gemini') return (data?.candidates?.[0]?.content?.parts ?? []).map((x) => x?.text ?? '').join('');
  return data?.choices?.[0]?.message?.content ?? '';
}

/** Models a provider offers (for the model field's suggestions). */
export async function listModels(id, { baseUrl, apiKey } = {}) {
  const p = provider(id);
  const base = trimUrl(baseUrl || p.baseUrl);
  if (p.kind === 'webgpu' || p.kind === 'none') return p.model ? [p.model] : [];
  if (p.kind === 'gemini') {
    if (!apiKey?.trim()) return [p.model];
    const res = await fetch(`${base}/models?key=${encodeURIComponent(apiKey.trim())}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return ((await res.json())?.models ?? []).map((m) => String(m.name || '').replace(/^models\//, '')).filter(Boolean);
  }
  if (p.kind === 'anthropic') {
    if (!apiKey?.trim()) return [p.model];
    const res = await fetch(`${base}/v1/models`, {
      headers: { 'x-api-key': apiKey.trim(), 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return ((await res.json())?.data ?? []).map((m) => String(m.id)).filter(Boolean);
  }
  try {
    const res = await fetch(`${base}/models`, { headers: apiKey?.trim() ? headersFor(p, apiKey) : undefined });
    if (res.ok) {
      const ids = ((await res.json())?.data ?? []).map((m) => String(m.id)).filter(Boolean);
      if (ids.length || p.id !== 'ollama') return ids;
    }
  } catch (err) {
    if (p.id !== 'ollama' && p.id !== 'custom') throw new Error(`Cannot reach ${base} (${err.message}; a local server must allow this page's origin)`);
  }
  // Ollama's own list.
  const res = await fetch(`${base.replace(/\/v1$/, '')}/api/tags`);
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${base}`);
  return ((await res.json())?.models ?? []).map((m) => String(m.name)).filter(Boolean);
}

// ---- Anthropic (official SDK) ---------------------------------------------------------------

let anthropicSdk = null;
// Models that take the server-side refusal fallback.
const FALLBACK_MODELS = /^claude-(fable-5|opus-5|sonnet-5-5)/;

async function askClaude(prompt, { baseUrl, model, apiKey }) {
  anthropicSdk ??= await import(ANTHROPIC_SDK_URL);
  const Anthropic = anthropicSdk.default ?? anthropicSdk.Anthropic;
  const base = trimUrl(baseUrl);
  const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true, ...(base && base !== 'https://api.anthropic.com' ? { baseURL: base } : {}) });
  const params = {
    model,
    max_tokens: 4096,
    output_config: { effort: 'low', format: { type: 'json_schema', schema: INTENTIONS_SCHEMA } },
    system: SYSTEM,
    messages: [{ role: 'user', content: prompt }],
  };
  // Refusal fallback: on a policy decline the API reruns on a fallback model.
  const response = FALLBACK_MODELS.test(model)
    ? await client.beta.messages.create({ ...params, betas: ['server-side-fallback-2026-07-01'], fallbacks: 'default' })
    : await client.messages.create(params);
  if (response.stop_reason === 'refusal') throw new Error('The model declined this prompt.');
  const text = response.content.find((b) => b.type === 'text')?.text;
  if (!text) throw new Error(`No answer (stop reason: ${response.stop_reason}).`);
  return { text, model: response.model };
}

// ---- WebGPU (in the browser) ----------------------------------------------------------------

const RUNTIMES = {
  lfm3: { file: 'Lfm3ForCausalLM.js', name: 'Lfm3ForCausalLM' },
  'lfm3-omni': { file: 'Lfm3OmniForConditionalGeneration.js', name: 'Lfm3OmniForConditionalGeneration' },
};

/** A WebGPU brain, loaded once (a large download: only when the person asks). */
class LocalBrain {
  constructor(id) {
    this.id = id;
    this.model = null;
    this.loading = null;
    this.queue = Promise.resolve();
  }

  get ready() {
    return Boolean(this.model);
  }

  load(onProgress = () => {}) {
    if (this.model) return Promise.resolve();
    this.loading ??= (async () => {
      try {
        if (!globalThis.navigator?.gpu) throw new Error('This browser has no WebGPU (try a recent Chrome or Edge, or Safari 26).');
        const p = provider(this.id);
        const rt = RUNTIMES[this.id];
        onProgress({ fraction: 0, message: 'Loading the runtime…' });
        const mod = await import(new URL(`../../vendor/${rt.file}`, import.meta.url).href);
        const cls = mod[rt.name];
        const check = await cls.checkAvailability(p.model).catch(() => true);
        if (check && typeof check === 'object' && check.ok === false) throw new Error(check.reason || 'This GPU cannot run the model.');
        this.model = await cls.from_pretrained(p.model, {
          onProgress: (e) => {
            const fraction = typeof e.progress === 'number' ? (e.progress > 1 ? e.progress / 100 : e.progress)
              : e.kind === 'bytes' && e.total ? e.loaded / e.total : typeof e.fraction === 'number' ? e.fraction : null;
            onProgress({ fraction, message: `Downloading ${p.label.split(' ·')[0]} (${p.size}, then cached)…` });
          },
        });
        onProgress({ fraction: 1, message: `${p.label.split(' ·')[0]} is ready on this machine.` });
      } catch (err) {
        this.loading = null;
        throw err;
      }
    })();
    return this.loading;
  }

  /** One answer, text only. Calls are queued (one generation at a time on the GPU). */
  complete(messages, { maxNewTokens = 1024 } = {}) {
    if (!this.model) return Promise.reject(new Error('Load the model first.'));
    const run = this.queue.then(async () => {
      const session = this.model.create_session();
      try {
        const opts = { max_new_tokens: maxNewTokens, do_sample: false };
        if (this.id === 'lfm3-omni') opts.generation_mode = 'text';
        const gen = session.generate(messages, opts);
        let text = '';
        for await (const ev of gen) if (ev.type === 'text' && ev.text) text += ev.text;
        const result = await gen.result;
        return text || result?.message?.content || '';
      } finally {
        await Promise.resolve(session.dispose?.()).catch(() => {});
      }
    });
    this.queue = run.catch(() => {});
    return run;
  }
}

const local = {};
export const localBrain = (id) => (local[id] ??= new LocalBrain(id));

// ---- one entry point -----------------------------------------------------------------------

/**
 * Read a prompt as intentions with the chosen brain.
 * settings: { provider, baseUrl, model, apiKey }
 * @returns {Promise<{ intentions: object, summary: string, model: string }>}
 */
export async function intentionsFromBrain(prompt, settings) {
  const p = provider(settings.provider);
  const model = settings.model?.trim() || p.model;
  let text;
  let used = model;
  if (p.kind === 'none') throw new Error('No brain chosen.');
  if (p.kind === 'webgpu') {
    const brain = localBrain(p.id);
    if (!brain.ready) throw new Error(`${p.label.split(' ·')[0]} is not loaded yet (press Load: ${p.size}).`);
    text = await brain.complete(messagesFor(prompt, false));
  } else if (p.kind === 'anthropic') {
    if (!settings.apiKey?.trim()) throw new Error('Claude needs an API key.');
    ({ text, model: used } = await askClaude(prompt, { ...settings, model }));
  } else {
    if (p.key && !settings.apiKey?.trim()) throw new Error(`${p.label} needs an API key.`);
    const s = { ...settings, model };
    // Structured output where it is known to work; otherwise (or if the server
    // rejects it) ask for JSON in words and parse leniently.
    const structured = p.id === 'openai' || p.id === 'lmstudio' || p.id === 'ollama' || p.kind === 'gemini';
    try {
      text = await postChat(p, s, messagesFor(prompt, false), structured);
    } catch (err) {
      if (!structured || err.status !== 400) throw err;
      text = await postChat(p, s, messagesFor(prompt, false), false);
    }
  }
  const { intentions, summary } = sanitize(extractJSON(text));
  return { intentions, summary, model: used };
}
