// Phrase memory: the improviser composes in sections rather than one
// movement at a time.
//
//   state      choose two to four movements freely and remember them as a
//              motif (A, B, C...)
//   develop    dance a remembered motif again, changed by one operation:
//              repeat, mirror, retrograde, fragment, accumulate (1; 1 2;
//              1 2 3), vary (swap one movement for its nearest relative),
//              slower or faster
//   digress    a short stretch of new, unremembered material
//
// Variety sets the balance: low variety develops and returns to its motifs;
// high variety states new ones and digresses more. A new prompt (different
// intentions) starts a new motif; earlier motifs stay in memory and can
// return later. All chance goes through the improviser's seeded random.

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';

/** The parts of the intentions that define "the same idea" (variety and transform weights do not). */
function ideaKey(I) {
  const { energy, level, size, travel, turn, armsUp, jump, stillness, focus, tags, avoid } = I;
  return JSON.stringify({ energy, level, size, travel, turn, armsUp, jump, stillness, focus, tags, avoid });
}

function pickWeighted(random, entries) {
  const total = entries.reduce((s, [, w]) => s + w, 0);
  let r = random() * total;
  for (const [value, w] of entries) {
    r -= w;
    if (r <= 0) return value;
  }
  return entries[entries.length - 1][0];
}

export class Composer {
  constructor() {
    this.motifs = []; // { name, key, items: [{ id, mirrored, retro }], uses }
    this.form = []; // section labels, in order
    this.plan = [];
    this.key = null;
    this.sinceNew = 0;
  }

  /** The next step: { item?, record?, role, phrase, tempo? } */
  next(improviser) {
    const I = improviser.intentions;
    const key = ideaKey(I);
    if (key !== this.key) {
      // A new idea: drop the rest of the current section and state a new motif.
      this.key = key;
      this.plan = [];
      this.forceNew = true;
    }
    // A motif whose movements were all forgotten cannot be developed.
    const known = (mo) => mo.items.length && mo.items.every((it) => improviser.movements.some((m) => m.id === it.id));
    while (!this.plan.length) this.plan = this.section(improviser, known);
    return this.plan.shift();
  }

  section(improviser, known) {
    const random = improviser.random;
    const v = improviser.intentions.variety ?? 0.5;
    const usable = this.motifs.filter((mo) => mo.items.length && known(mo));
    const fitting = usable.filter((mo) => mo.key === this.key);
    const pNew = this.forceNew || !fitting.length ? 1 : Math.min(0.9, 0.12 + 0.45 * v + 0.08 * this.sinceNew);
    this.forceNew = false;

    if (random() < pNew) {
      // State a new motif.
      const name = LETTERS[this.motifs.length % LETTERS.length] + (this.motifs.length >= LETTERS.length ? Math.floor(this.motifs.length / LETTERS.length) + 1 : '');
      const length = 2 + Math.floor(random() * (v < 0.4 ? 2 : 3)); // 2-3, or 2-4
      const motif = { name, key: this.key, items: [], uses: 0 };
      this.motifs.push(motif);
      this.form.push(name);
      this.sinceNew = 0;
      return Array.from({ length }, (_, i) => ({
        record: (item) => motif.items.push(item),
        role: `stating ${name} (${i + 1}/${length})`,
        phrase: name,
      }));
    }

    this.sinceNew++;
    // Develop the latest fitting motif mostly; sometimes return to an older one.
    const latest = fitting[fitting.length - 1];
    const motif = fitting.length > 1 && random() < 0.3 ? fitting[Math.floor(random() * (fitting.length - 1))] : latest;
    motif.uses++;
    const returning = motif !== latest;
    const items = motif.items;
    const L = items.length;
    const op = pickWeighted(random, [
      ['repeat', 1.2 * (1 - v) + 0.1],
      ['mirror', 1],
      ['retrograde', 0.6],
      ['fragment', L > 1 ? 0.7 : 0],
      ['accumulate', L > 1 ? 0.3 + 0.7 * (1 - v) : 0],
      ['vary', 0.5 + 0.7 * v],
      ['slower', 0.4],
      ['faster', 0.4],
      ['digress', 0.3 + 0.9 * v],
    ]);
    const n = motif.name;
    const steps = (list, label, extra = {}) =>
      list.map((item, i) => ({ item, role: `${returning ? 'returning to ' : ''}${label} (${i + 1}/${list.length})`, phrase: label, ...extra }));

    let label;
    let plan;
    switch (op) {
      case 'repeat':
        label = `${n} again`;
        plan = steps(items, label);
        break;
      case 'mirror':
        label = `${n} mirrored`;
        plan = steps(items.map((it) => ({ ...it, mirrored: !it.mirrored })), label);
        break;
      case 'retrograde':
        label = `${n} backwards`;
        plan = steps([...items].reverse().map((it) => ({ ...it, retro: !it.retro })), label);
        break;
      case 'fragment': {
        const k = 1 + Math.floor(random() * (L - 1));
        label = `fragment of ${n}`;
        plan = steps(items.slice(0, k), label);
        break;
      }
      case 'accumulate': {
        const seq = [];
        for (let k = 1; k <= L && seq.length + k <= 9; k++) seq.push(...items.slice(0, k));
        label = `${n} accumulating`;
        plan = steps(seq, label);
        break;
      }
      case 'vary': {
        const k = Math.floor(random() * L);
        label = `${n} varied`;
        plan = steps(items.map((it, i) => (i === k ? { ...it, vary: true } : it)), label);
        break;
      }
      case 'slower':
        label = `${n} slower`;
        plan = steps(items, label, { tempo: 0.75 });
        break;
      case 'faster':
        label = `${n} faster`;
        plan = steps(items, label, { tempo: 1.3 });
        break;
      default: {
        const length = 1 + Math.floor(random() * 2);
        label = 'digression';
        plan = Array.from({ length }, (_, i) => ({ role: `digression (${i + 1}/${length})`, phrase: label }));
      }
    }
    this.form.push(returning && op !== 'digress' ? `↩ ${label}` : label);
    return plan;
  }
}
