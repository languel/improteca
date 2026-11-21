// Brains: request shapes per provider, lenient JSON reading, and a round trip
// through a fake OpenAI-compatible server (including one that rejects
// structured output).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { PROVIDERS, provider, buildRequest, extractJSON, sanitize, intentionsFromBrain, listModels, messagesFor } from '../src/dancer/brains.js';

test('every provider has what the page needs', () => {
  const ids = PROVIDERS.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const p of PROVIDERS) {
    if (p.kind === 'none') continue;
    assert.ok(p.model, `${p.id} has a default model`);
    if (p.kind !== 'webgpu') assert.match(p.baseUrl, /^https?:\/\//);
  }
  assert.equal(provider('nope').id, 'none');
});

test('OpenAI-compatible and Gemini requests are shaped right', () => {
  const messages = messagesFor('slow', false);
  const o = buildRequest(provider('openrouter'), { baseUrl: 'https://openrouter.ai/api/v1/', model: 'm', apiKey: 'k', messages, json: true });
  assert.equal(o.url, 'https://openrouter.ai/api/v1/chat/completions');
  assert.equal(o.headers.Authorization, 'Bearer k');
  assert.equal(o.headers['X-Title'], 'improteca');
  assert.equal(o.body.response_format.json_schema.strict, true);
  const l = buildRequest(provider('ollama'), { model: 'lfm3', messages });
  assert.equal(l.url, 'http://localhost:11434/v1/chat/completions');
  assert.equal(l.headers.Authorization, undefined);
  assert.equal(l.body.response_format, undefined);
  const g = buildRequest(provider('gemini'), { model: 'models/gemini-2.5-flash', apiKey: 'k k', messages, json: true });
  assert.equal(g.url, 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=k%20k');
  assert.equal(g.headers.Authorization, undefined);
  assert.ok(g.body.systemInstruction.parts[0].text.includes('JSON'));
  assert.deepEqual(g.body.contents, [{ role: 'user', parts: [{ text: 'slow' }] }]);
  assert.equal(g.body.generationConfig.responseMimeType, 'application/json');
});

test('JSON is found in thinking, prose and code fences', () => {
  assert.deepEqual(extractJSON('<think>{"no": 1}</think>Sure! ```json\n{"a": {"b": "}"}}\n``` done'), { a: { b: '}' } });
  assert.throws(() => extractJSON('no json here'));
  assert.throws(() => extractJSON('{"a": 1'));
});

test('whatever comes back becomes valid intentions', () => {
  const { intentions, summary } = sanitize({ energy: 3, level: -1, size: 'big', focus: 'tail', tags: ['ballet', 'ballet', 'jazz'], tempo: 9, variety: 0.2, extra: 1, summary: 'ok' });
  assert.equal(intentions.energy, 1);
  assert.equal(intentions.level, 0);
  assert.equal(intentions.size, null);
  assert.equal(intentions.focus, null);
  assert.deepEqual(intentions.tags, ['ballet']);
  assert.equal(intentions.tempo, 1.4);
  assert.equal(intentions.variety, 0.2);
  assert.equal(intentions.extra, undefined);
  assert.equal(summary, 'ok');
});

test('a local OpenAI-compatible server reads a prompt, with or without structured output', async () => {
  const seen = [];
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      res.setHeader('Content-Type', 'application/json');
      if (req.url === '/v1/models') return res.end(JSON.stringify({ data: [{ id: 'tiny' }, { id: 'big' }] }));
      const json = JSON.parse(body);
      seen.push(json);
      if (json.response_format) {
        res.statusCode = 400;
        return res.end('{"error":"response_format not supported"}');
      }
      res.end(JSON.stringify({ choices: [{ message: { content: 'Here you go: {"energy":0.1,"level":0.2,"size":null,"travel":null,"turn":null,"armsUp":null,"jump":0,"stillness":0.7,"focus":"arms","tags":[],"avoid":["everyday"],"tempo":0.7,"variety":0.5,"mirror":0.35,"retrograde":0.1,"summary":"Sinking, slow, arms lead."}' } }] }));
    });
  });
  await new Promise((r) => server.listen(0, r));
  const baseUrl = `http://127.0.0.1:${server.address().port}/v1`;
  try {
    assert.deepEqual(await listModels('lmstudio', { baseUrl }), ['tiny', 'big']);
    const out = await intentionsFromBrain('slow and low, like sinking into water', { provider: 'lmstudio', baseUrl, model: 'tiny' });
    assert.equal(seen.length, 2, 'retried without structured output after a 400');
    assert.equal(seen[1].model, 'tiny');
    assert.equal(out.intentions.focus, 'arms');
    assert.equal(out.intentions.tempo, 0.7);
    assert.deepEqual(out.intentions.avoid, ['everyday']);
    assert.equal(out.summary, 'Sinking, slow, arms lead.');
  } finally {
    server.close();
  }
});

test('cloud brains ask for a key before calling out', async () => {
  await assert.rejects(intentionsFromBrain('x', { provider: 'openai' }), /API key/);
  await assert.rejects(intentionsFromBrain('x', { provider: 'anthropic' }), /API key/);
  // A WebGPU brain loads on demand; without WebGPU (Node) that fails clearly.
  await assert.rejects(intentionsFromBrain('x', { provider: 'lfm3' }), /WebGPU/);
});
