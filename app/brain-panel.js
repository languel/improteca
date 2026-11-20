// The Brain panel: choose a language model for reading prompts, load a WebGPU
// one, set a local server or a cloud key. Settings stay on this device; keys
// stay in memory unless the person asks to remember them.

import { PROVIDERS, provider, listModels, localBrain, intentionsFromBrain } from '../src/dancer/brains.js';
import { describeIntentions } from '../src/dancer/intentions.js';

const $ = (id) => document.getElementById(id);
const STORE = 'improteca.brain';
const KEYS = 'improteca.brain-keys';
const OLD_KEY = 'improteca.anthropic-key';

const load = (k, fallback) => {
  try {
    return JSON.parse(localStorage.getItem(k)) ?? fallback;
  } catch {
    return fallback;
  }
};
const save = (k, v) => {
  try {
    if (v === null) localStorage.removeItem(k);
    else localStorage.setItem(k, JSON.stringify(v));
  } catch {
    // Storage blocked: settings last for this visit.
  }
};

const settings = load(STORE, { provider: 'none', per: {} });
settings.per ??= {};
const keys = load(KEYS, {}); // remembered keys only
const session = { ...keys }; // keys in memory for this visit
// The earlier Claude-only key.
try {
  const old = localStorage.getItem(OLD_KEY);
  if (old) {
    keys.anthropic = session.anthropic = old;
    if (settings.provider === 'none') settings.provider = 'anthropic';
    save(KEYS, keys);
    localStorage.removeItem(OLD_KEY);
  }
} catch {
  // ignore
}

/** The current brain settings, for intentionsFromBrain. */
export function brainSettings() {
  const p = provider(settings.provider);
  const per = settings.per[p.id] ?? {};
  return { provider: p.id, baseUrl: per.baseUrl || p.baseUrl, model: per.model || p.model, apiKey: session[p.id] || '' };
}

/** A short name for the chosen brain, for notes. */
export function brainName() {
  const p = provider(settings.provider);
  return p.kind === 'none' ? '' : p.label.split(' ·')[0].split(' (')[0];
}

export function brainActive() {
  return provider(settings.provider).kind !== 'none';
}

function status(text) {
  $('brain-status').textContent = text;
}

function render() {
  const p = provider(settings.provider);
  const per = settings.per[p.id] ?? {};
  $('brain').value = p.id;
  $('brain-local').hidden = p.kind !== 'webgpu';
  $('brain-remote').hidden = p.kind === 'none' || p.kind === 'webgpu';
  $('brain-test').hidden = p.kind === 'none';
  if (p.kind === 'webgpu') {
    const b = localBrain(p.id);
    $('brain-load').textContent = b.ready ? 'Loaded' : 'Load';
    $('brain-load').disabled = b.ready;
    $('brain-size').textContent = b.ready ? 'Runs on this machine.' : `Downloads ${p.size} once from Hugging Face, then cached.`;
  } else if (p.kind !== 'none') {
    $('brain-url').value = per.baseUrl || p.baseUrl;
    $('brain-model').value = per.model || p.model;
    $('brain-key-row').querySelector('input').placeholder = p.key ? 'your key' : 'optional';
    $('api-key').value = session[p.id] || '';
    $('remember-key').checked = Boolean(keys[p.id]);
  }
  if (p.kind === 'none') status('Prompts are read by the built-in movement vocabulary.');
  else if (p.kind === 'webgpu') status(localBrain(p.id).ready ? '' : 'Not loaded yet.');
  else status('');
}

function persist() {
  save(STORE, settings);
}

export function setupBrainPanel() {
  $('brain').replaceChildren(...PROVIDERS.map((p) => new Option(p.label, p.id)));
  render();
  $('brain').addEventListener('change', () => {
    settings.provider = $('brain').value;
    persist();
    render();
  });
  const field = (id, key) => $(id).addEventListener('change', () => {
    const p = provider(settings.provider);
    settings.per[p.id] = { ...(settings.per[p.id] ?? {}), [key]: $(id).value.trim() };
    persist();
  });
  field('brain-url', 'baseUrl');
  field('brain-model', 'model');
  const rememberKey = () => {
    const id = settings.provider;
    session[id] = $('api-key').value.trim();
    if ($('remember-key').checked && session[id]) keys[id] = session[id];
    else delete keys[id];
    save(KEYS, Object.keys(keys).length ? keys : null);
  };
  $('api-key').addEventListener('input', rememberKey);
  $('remember-key').addEventListener('change', rememberKey);

  $('brain-load').addEventListener('click', async () => {
    const p = provider(settings.provider);
    $('brain-load').disabled = true;
    try {
      await localBrain(p.id).load(({ fraction, message }) => {
        status(fraction === null || fraction === undefined ? message : `${message} ${Math.round(fraction * 100)}%`);
      });
      render();
      status(`${brainName()} is ready on this machine.`);
    } catch (err) {
      console.error(err);
      $('brain-load').disabled = false;
      status(`Could not load: ${err.message || err}`);
    }
  });

  $('brain-list').addEventListener('click', async () => {
    const s = brainSettings();
    status('Asking for the list of models…');
    try {
      const ids = await listModels(s.provider, s);
      $('brain-models').replaceChildren(...ids.map((id) => new Option(id, id)));
      status(ids.length ? `${ids.length} models: pick one in the Model field.` : 'No models listed.');
    } catch (err) {
      status(`Could not list models: ${err.message || err}`);
    }
  });

  $('brain-test').addEventListener('click', async () => {
    const prompt = document.getElementById('prompt').value.trim() || 'like seaweed in a slow current';
    status(`Reading "${prompt}"…`);
    const t0 = performance.now();
    try {
      const { intentions, summary, model } = await intentionsFromBrain(prompt, brainSettings());
      const words = describeIntentions(intentions);
      status(`${model} (${((performance.now() - t0) / 1000).toFixed(1)} s): ${summary || '—'} → ${words.join(', ') || 'no preferences'}`);
    } catch (err) {
      status(`Did not work: ${err.message || err}`);
    }
  });
}
