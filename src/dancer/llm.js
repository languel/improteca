// Prompt -> intentions with Claude (optional; the lexicon in intentions.js
// always works without it). Runs in the browser with the user's own API key,
// which is sent only to Anthropic and kept in this tab's memory unless the
// user chooses to remember it.

import { NEUTRAL } from './improviser.js';

export const SDK_URL = 'https://cdn.jsdelivr.net/npm/@anthropic-ai/sdk@1.6.0/+esm';
export const MODEL = 'claude-opus-5-5';

const quality = (description) => ({ type: ['number', 'null'], description: `${description} 0 to 1, or null for no preference.` });

/** JSON schema for the intentions object (structured outputs: all keys required, no extras). */
export const INTENTIONS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['energy', 'level', 'size', 'travel', 'turn', 'armsUp', 'jump', 'stillness', 'focus', 'tags', 'avoid', 'tempo', 'variety', 'mirror', 'retrograde', 'summary'],
  properties: {
    energy: quality('Calm (0) to vigorous (1).'),
    level: quality('Low, near the floor (0) to high, lifted (1).'),
    size: quality('Small and close to the body (0) to large and extended (1).'),
    travel: quality('On the spot (0) to travelling across the space (1).'),
    turn: quality('No turning (0) to turning and spinning (1).'),
    armsUp: quality('Arms low (0) to arms overhead (1).'),
    jump: quality('Grounded (0) to jumping, airborne (1).'),
    stillness: quality('Continuous flow (0) to pauses and holds (1).'),
    focus: { type: ['string', 'null'], enum: ['handL', 'handR', 'footL', 'footR', 'head', 'arms', 'legs', null], description: 'Body part that should lead, or null.' },
    tags: { type: 'array', items: { type: 'string', enum: ['ballet', 'dance', 'everyday'] }, description: 'Preferred material: ballet, dance (contemporary/modern), everyday (pedestrian gestures).' },
    avoid: { type: 'array', items: { type: 'string', enum: ['ballet', 'dance', 'everyday'] }, description: 'Material to avoid.' },
    tempo: { type: 'number', description: 'Playback tempo: 1 natural, 0.6 much slower, 1.4 much faster.' },
    variety: { type: 'number', description: 'Repetitive (0) to unpredictable (1); 0.5 default.' },
    mirror: { type: 'number', description: 'Weight for mirrored (left/right swapped) versions; 0.35 default.' },
    retrograde: { type: 'number', description: 'Weight for movements danced backwards; 0.1 default.' },
    summary: { type: 'string', description: 'One short sentence: how you read the prompt as movement.' },
  },
};

const SYSTEM = `You translate a choreographer's prompt into intentions for an improvising virtual dancer.
The dancer knows recorded movements (ballet, contemporary dance and everyday gestures from motion capture)
and chooses among them by these qualities. Read imagery and metaphor as movement qualities
(e.g. "like seaweed" = continuous, low energy, arms leading; "a storm" = vigorous, travelling, turning).
Set a quality only when the prompt implies it; otherwise null. Keep tempo between 0.6 and 1.4.`;

/** Load the SDK lazily (only when the user turns the language model on). */
let sdk = null;
async function client(apiKey) {
  sdk ??= await import(SDK_URL);
  const Anthropic = sdk.default ?? sdk.Anthropic;
  return new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
}

/**
 * @returns {Promise<{ intentions: object, summary: string, model: string }>}
 */
export async function intentionsFromClaude(prompt, apiKey) {
  const c = await client(apiKey);
  const response = await c.beta.messages.create({
    model: MODEL,
    max_tokens: 4096,
    // Refusal fallback: on a policy decline the API reruns on a fallback model.
    betas: ['server-side-fallback-2026-07-01'],
    fallbacks: 'default',
    output_config: { effort: 'low', format: { type: 'json_schema', schema: INTENTIONS_SCHEMA } },
    system: SYSTEM,
    messages: [{ role: 'user', content: prompt }],
  });
  if (response.stop_reason === 'refusal') throw new Error('The model declined this prompt.');
  const text = response.content.find((b) => b.type === 'text')?.text;
  if (!text) throw new Error(`No answer (stop reason: ${response.stop_reason}).`);
  const { summary, ...rest } = JSON.parse(text);
  return { intentions: { ...NEUTRAL, ...rest }, summary, model: response.model };
}
