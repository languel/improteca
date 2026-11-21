// Prompt -> intentions, without a model.
//
// A small lexicon of movement words (Laban-flavoured and everyday) maps a
// prompt to the improviser's intention values, with intensifiers ("very"),
// softeners ("a bit") and negation ("not", "no", "less", "without"). It
// explains what it understood. An optional language model can produce the
// same object from freer text (see brains.js).

import { NEUTRAL } from './improviser.js';

// word -> [quality, value]; value 0..1 for qualities, or a special handler.
const LEXICON = [
  // energy
  [/\b(calm|gentle|soft|quiet|tender|serene|peaceful|lazy|tired|sleepy|relaxed|light(?:ly)?|float(?:ing|y)?)\b/, 'energy', 0.1],
  [/\b(energetic(?:ally)?|vigorous(?:ly)?|wild(?:ly)?|explosive|powerful|strong|fierce|frantic|excited|angry|intense|sharp|punchy|dynamic)\b/, 'energy', 0.9],
  // level
  [/\b(low|floor|ground|crouch(?:ing)?|sink(?:ing)?|deep|grounded|earthy|kneel(?:ing)?)\b/, 'level', 0.05],
  [/\b(high|tall|lift(?:ed|ing)?|rise|rising|reach(?:ing)? up|upward|elevated|on (?:your |her |his |their )?toes|relev[eé])\b/, 'level', 0.95],
  // size
  [/\b(small|tiny|close|contained|intimate|minimal|subtle|narrow)\b/, 'size', 0.1],
  [/\b(big|large|expansive|wide|open|huge|grand|broad|extended|stretch(?:ed|ing|y)?)\b/, 'size', 0.9],
  // travel
  [/\b(in place|on the spot|stationary|stay(?:ing)? put|rooted)\b/, 'travel', 0.05],
  [/\b(travel(?:ling|ing)?|move around|across the (?:stage|space|room)|walk(?:ing)?|run(?:ning)?|roam(?:ing)?|wander(?:ing)?)\b/, 'travel', 0.9],
  // turning
  [/\b(turn(?:s|ing)?|spin(?:s|ning)?|pirouettes?|twirl(?:s|ing)?|whirl(?:ing)?|rotat(?:e|ing|ion)|spiral(?:s|ing)?)\b/, 'turn', 0.95],
  // arms
  [/\b(arms? (?:up|high|overhead|raised)|reach(?:ing)? (?:up|high)|hands? (?:up|overhead)|port de bras)\b/, 'armsUp', 0.95],
  // jumps
  [/\b(jump(?:s|ing|y)?|leap(?:s|ing)?|hop(?:s|ping)?|bounc(?:e|y|ing)|jet[eé]s?|airborne|fly(?:ing)?)\b/, 'jump', 0.95],
  // stillness / continuity
  [/\b(still|stillness|pause(?:s|d)?|hold(?:s|ing)?|sustained|slow(?:ly)?|suspend(?:ed)?|balance(?:d|s)?|meditative)\b/, 'stillness', 0.85],
  [/\b(flow(?:ing)?|continuous|fluid|smooth|non-?stop|restless)\b/, 'stillness', 0.1],
];

const TEMPO = [
  [/\b(very slow(?:ly)?|glacial|in slow motion)\b/, 0.6],
  [/\b(slow(?:ly|er)?|unhurried|leisurely|sustained)\b/, 0.8],
  [/\b(very fast|frantic|rapid(?:ly)?|double time)\b/, 1.4],
  [/\b(fast(?:er)?|quick(?:ly|er)?|brisk|hurried|sudden)\b/, 1.2],
];

const FOCUS = [
  [/\b(left hand|left arm)\b/, 'handL'],
  [/\b(right hand|right arm)\b/, 'handR'],
  [/\b(left (?:foot|leg))\b/, 'footL'],
  [/\b(right (?:foot|leg))\b/, 'footR'],
  [/\b(head|gaze|neck|look(?:ing)?)\b/, 'head'],
  [/\b(arms|hands|upper body|port de bras)\b/, 'arms'],
  [/\b(legs|feet|footwork|lower body)\b/, 'legs'],
];

const MATERIAL = [
  [/\b(ballet|classical|balletic)\b/, 'ballet'],
  [/\b(contemporary|modern)\b/, 'dance'],
  [/\b(everyday|pedestrian|ordinary|daily|casual|natural)\b/, 'everyday'],
  [/\b(films?|archival|vintage|historic(?:al)?|old footage|silent film|muybridge|sandow)\b/, 'film'],
];

const VARIETY = [
  [/\b(repeat(?:ing|edly|ed)?|repetitive|same|again|loop(?:ing)?|minimal(?:ist)?|ritual)\b/, 0.1],
  [/\b(surprise|surprising|unpredictable|random|chaotic|varied|eclectic|improvis(?:e|ed|ing))\b/, 0.9],
];

const NEGATION = /\b(not|no|never|without|less|avoid|don't|do not|stop)\b(?:\s+\w+){0,2}\s*$/;
const STRONG = /\b(very|really|extremely|so|super|completely|totally)\s*$/;
const WEAK = /\b(a bit|slightly|somewhat|a little|kind of|rather)\s*$/;

/** Returns { intentions, understood: string[] }. */
export function parsePrompt(prompt) {
  // Clause punctuation becomes a barrier that negation does not cross.
  const text = ` ${prompt.toLowerCase().replace(/[,.;:!?()]/g, ' | ').replace(/[^\p{L}\s'|-]/gu, ' ').replace(/\s+/g, ' ')} `;
  const intentions = { ...NEUTRAL, tags: [], avoid: [] };
  const understood = [];
  const prefix = (index) => text.slice(Math.max(0, index - 30), index);

  for (const [re, quality, value] of LEXICON) {
    const g = new RegExp(re.source, 'g');
    for (const match of text.matchAll(g)) {
      const before = prefix(match.index);
      let v = value;
      if (NEGATION.test(before)) v = 1 - value;
      else if (STRONG.test(before)) v = value > 0.5 ? 1 : 0;
      else if (WEAK.test(before)) v = 0.5 + (value - 0.5) * 0.5;
      intentions[quality] = v;
      understood.push(`${match[0].trim()} → ${quality} ${v.toFixed(2)}`);
    }
  }
  for (const [re, tempo] of TEMPO) {
    const m = text.match(re);
    if (m) {
      intentions.tempo = NEGATION.test(prefix(m.index)) ? 1 / tempo : tempo;
      understood.push(`${m[0].trim()} → tempo ×${intentions.tempo.toFixed(2)}`);
      break;
    }
  }
  for (const [re, part] of FOCUS) {
    const m = text.match(re);
    if (m) {
      intentions.focus = part;
      understood.push(`${m[0].trim()} → led by ${part}`);
      break;
    }
  }
  for (const [re, tag] of MATERIAL) {
    const m = text.match(re);
    if (!m) continue;
    if (NEGATION.test(prefix(m.index))) {
      intentions.avoid.push(tag);
      understood.push(`${m[0].trim()} → avoid ${tag}`);
    } else {
      intentions.tags.push(tag);
      understood.push(`${m[0].trim()} → ${tag} material`);
    }
  }
  for (const [re, variety] of VARIETY) {
    const m = text.match(re);
    if (m) {
      intentions.variety = variety;
      understood.push(`${m[0].trim()} → variety ${variety}`);
      break;
    }
  }
  if (/\b(mirror(?:ed)?|other side|symmetr\w*)\b/.test(text)) {
    intentions.mirror = 0.8;
    understood.push('mirror → more mirrored versions');
  }
  if (/\b(backwards?|reverse(?:d)?|retrograde|rewind)\b/.test(text)) {
    intentions.retrograde = 0.7;
    understood.push('backwards → more retrograde');
  }
  return { intentions, understood };
}

/** A short human summary of intentions for display. */
export function describeIntentions(I) {
  const words = {
    energy: ['calm', 'vigorous'], level: ['low', 'high'], size: ['small', 'expansive'], travel: ['on the spot', 'travelling'],
    turn: ['no turns', 'turning'], armsUp: ['arms low', 'arms high'], jump: ['grounded', 'jumping'], stillness: ['flowing', 'with pauses'],
  };
  const out = [];
  // A middling value (0.35 to 0.65) is no strong preference either way: no word for it.
  for (const [k, [lo, hi]] of Object.entries(words)) {
    const v = I[k];
    if (v === null || v === undefined) continue;
    if (v >= 0.65) out.push(hi);
    else if (v <= 0.35) out.push(lo);
  }
  if (I.focus) out.push(`led by ${({ handL: 'left hand', handR: 'right hand', footL: 'left foot', footR: 'right foot', head: 'head', arms: 'arms', legs: 'legs' })[I.focus]}`);
  const tag = (t) => ({ dance: 'contemporary', film: 'film material' })[t] || t;
  if (I.tags?.length) out.push(I.tags.map(tag).join(' + '));
  for (const t of I.avoid || []) out.push(`no ${tag(t)}`);
  if (I.tempo && Math.abs(I.tempo - 1) > 0.05) out.push(I.tempo > 1 ? 'faster' : 'slower');
  if (I.variety <= 0.2) out.push('repetitive');
  if (I.variety >= 0.8) out.push('unpredictable');
  return out;
}
