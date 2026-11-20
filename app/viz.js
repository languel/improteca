// Visualisations of the moving body, functional and artistic.
//
//   trails      a line per chosen body part (hands, feet, head, pelvis), up to
//               minutes long, fading toward the past
//   ribbons     surfaces swept by a forearm or shin (elbow-hand, knee-ankle)
//   echoes      frozen copies of the figure left behind at intervals, fading
//   footprints  marks where the constraint layer planted a foot
//   activity    markers on joints the constraints corrected this frame
//
// Everything is fed with world poses through update(local, info) once per
// played frame.

import * as THREE from 'three';
import { TF } from '../src/engine/math.js';
import { makeFigure } from './figures.js';

export const PARTS = {
  handL: { joint: 'LeftHand', label: 'Left hand', color: '#d1495b' },
  handR: { joint: 'RightHand', label: 'Right hand', color: '#00798c' },
  footL: { joint: 'LeftToeBase', label: 'Left foot', color: '#edae49' },
  footR: { joint: 'RightToeBase', label: 'Right foot', color: '#66a182' },
  head: { joint: 'HeadEnd', label: 'Head', color: '#8d6cab' },
  hips: { joint: 'Hips', label: 'Pelvis', color: '#6d675d' },
};
const RIBBONS = {
  forearmL: ['LeftForeArm', 'LeftHand'],
  forearmR: ['RightForeArm', 'RightHand'],
  shinL: ['LeftShin', 'LeftFoot'],
  shinR: ['RightShin', 'RightFoot'],
};
const FPS = 30;

/** A long line kept in a ring buffer, with colours fading toward the past. */
class Trail {
  constructor(scene, color, capacity) {
    this.scene = scene;
    this.color = new THREE.Color(color);
    this.resize(capacity);
  }
  resize(capacity) {
    this.dispose();
    this.capacity = Math.max(2, capacity);
    this.points = new Float32Array(this.capacity * 3);
    this.count = 0;
    this.head = 0;
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.capacity * 3), 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.capacity * 3), 3));
    this.material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, depthWrite: false });
    this.line = new THREE.Line(this.geometry, this.material);
    this.line.frustumCulled = false;
    this.scene.add(this.line);
  }
  push(x, y, z) {
    this.points.set([x, y, z], this.head * 3);
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
  }
  /** Write the buffer oldest-first, fading toward `background`. */
  render(background) {
    const pos = this.geometry.attributes.position.array;
    const col = this.geometry.attributes.color.array;
    const n = this.count;
    const start = (this.head - n + this.capacity) % this.capacity;
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const k = ((start + i) % this.capacity) * 3;
      pos[i * 3] = this.points[k];
      pos[i * 3 + 1] = this.points[k + 1];
      pos[i * 3 + 2] = this.points[k + 2];
      c.copy(background).lerp(this.color, 0.15 + 0.85 * ((i + 1) / n) ** 0.8);
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    }
    this.geometry.setDrawRange(0, n);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
  }
  clear() {
    this.count = 0;
    this.head = 0;
    this.geometry?.setDrawRange(0, 0);
  }
  dispose() {
    if (!this.line) return;
    this.scene.remove(this.line);
    this.geometry.dispose();
    this.material.dispose();
    this.line = null;
  }
}

/**
 * A swept limb: the segment between two joints (e.g. elbow and hand) drawn
 * as a line every few frames, like hatching, fading toward the past.
 */
class Ribbon {
  constructor(scene, color, capacity, every = 2) {
    this.scene = scene;
    this.color = new THREE.Color(color);
    this.capacity = Math.max(2, Math.ceil(capacity / every));
    this.every = every;
    this.tick = 0;
    this.a = new Float32Array(this.capacity * 3);
    this.b = new Float32Array(this.capacity * 3);
    this.count = 0;
    this.head = 0;
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.capacity * 2 * 3), 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(this.capacity * 2 * 3), 3));
    this.material = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.7, depthWrite: false });
    this.mesh = new THREE.LineSegments(this.geometry, this.material);
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);
  }
  push(a, b) {
    if (this.tick++ % this.every) return;
    this.a.set(a, this.head * 3);
    this.b.set(b, this.head * 3);
    this.head = (this.head + 1) % this.capacity;
    this.count = Math.min(this.count + 1, this.capacity);
  }
  render(background) {
    const pos = this.geometry.attributes.position.array;
    const col = this.geometry.attributes.color.array;
    const n = this.count;
    const start = (this.head - n + this.capacity) % this.capacity;
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const k = ((start + i) % this.capacity) * 3;
      pos.set([this.a[k], this.a[k + 1], this.a[k + 2]], i * 6);
      pos.set([this.b[k], this.b[k + 1], this.b[k + 2]], i * 6 + 3);
      c.copy(background).lerp(this.color, 0.6 * ((i + 1) / n) ** 1.5);
      col.set([c.r, c.g, c.b, c.r, c.g, c.b], i * 6);
    }
    this.geometry.setDrawRange(0, n * 2);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
  }
  clear() {
    this.count = 0;
    this.head = 0;
    this.geometry.setDrawRange(0, 0);
  }
  dispose() {
    this.scene.remove(this.mesh);
    this.geometry.dispose();
    this.material.dispose();
  }
}

export const DEFAULT_VIEW = Object.freeze({
  figure: 'solid',
  trails: ['handL', 'handR', 'footL', 'footR'],
  trailSeconds: 8,
  ribbons: false,
  echoEvery: 0, // seconds; 0 = off
  echoCount: 6,
  echoFigure: 'same',
  footprints: true,
  activity: false,
  monochrome: false, // trails in the figure's colour
  background: 'auto', // 'auto' | 'light' | 'dark'
});

export const PRESETS = {
  studio: { label: 'Studio', view: { figure: 'solid', trails: ['hips'], trailSeconds: 12, ribbons: false, echoEvery: 0, footprints: false, activity: false, background: 'auto' } },
  functional: { label: 'Functional', view: { figure: 'sticks', trails: ['handL', 'handR', 'footL', 'footR', 'head', 'hips'], trailSeconds: 6, ribbons: false, echoEvery: 0, footprints: true, activity: true, background: 'auto' } },
  drawn: { label: 'Drawn (after Ghostcatching)', view: { figure: 'drawn', trails: ['handL', 'handR', 'footL', 'footR', 'head'], trailSeconds: 30, ribbons: true, echoEvery: 2, echoCount: 5, echoFigure: 'drawn', footprints: false, activity: false, monochrome: true, background: 'dark' } },
  echoes: { label: 'Time echoes', view: { figure: 'solid', trails: ['handL', 'handR'], trailSeconds: 4, ribbons: false, echoEvery: 0.5, echoCount: 12, echoFigure: 'same', footprints: false, activity: false, background: 'auto' } },
  body: { label: 'Body', view: { figure: 'body', trails: [], trailSeconds: 8, ribbons: false, echoEvery: 0, footprints: false, activity: false, background: 'auto' } },
};

export class Viz {
  constructor(stage, skeleton) {
    this.stage = stage;
    this.sk = skeleton;
    this.view = { ...DEFAULT_VIEW };
    this.trails = {};
    this.ribbons = {};
    this.echoes = [];
    this.sinceEcho = 0;
    this.footprints = [];
    this.world = new Float64Array(skeleton.numJoints * TF);
    this.figure = null;
    this.figureStyle = null;
    this.color = '#2f2c28';
    this.lastLocal = null;
    // Markers for corrected joints.
    this.marks = new THREE.Group();
    stage.scene.add(this.marks);
    this.markGeom = new THREE.SphereGeometry(0.035, 12, 8);
    this.markMat = new THREE.MeshBasicMaterial({ color: '#e0a100', transparent: true, opacity: 0.85, depthTest: false });
    this.footGeom = new THREE.CircleGeometry(0.06, 20).rotateX(-Math.PI / 2).scale(0.8, 1, 1.6);
  }

  /** Apply a (partial) view; rebuilds what changed. */
  async set(view) {
    this.view = { ...this.view, ...view };
    const v = this.view;
    this.stage.setBackground(v.background);
    if (v.figure !== this.figureStyle) {
      const fig = await makeFigure(v.figure, this.stage, this.sk, { color: this.figureColor() });
      this.figure?.dispose();
      this.figure = fig;
      this.figureStyle = v.figure;
      if (this.lastLocal) fig.setPose(this.lastLocal);
    } else this.figure?.setColor(this.figureColor());
    const cap = Math.round(v.trailSeconds * FPS);
    const tone = (c) => (v.monochrome ? this.figureColor() : c);
    for (const key of Object.keys(PARTS)) {
      const want = v.trails.includes(key);
      if (want && (!this.trails[key] || this.trails[key].capacity !== cap)) {
        this.trails[key]?.dispose();
        this.trails[key] = new Trail(this.stage.scene, tone(PARTS[key].color), cap);
      } else if (!want && this.trails[key]) {
        this.trails[key].dispose();
        delete this.trails[key];
      }
    }
    for (const key of Object.keys(RIBBONS)) {
      if (v.ribbons && (!this.ribbons[key] || this.ribbons[key].capacity !== cap)) {
        this.ribbons[key]?.dispose();
        const color = key.endsWith('L') ? (key.startsWith('fore') ? PARTS.handL.color : PARTS.footL.color) : key.startsWith('fore') ? PARTS.handR.color : PARTS.footR.color;
        this.ribbons[key] = new Ribbon(this.stage.scene, tone(color), Math.min(cap, 300));
      } else if (!v.ribbons && this.ribbons[key]) {
        this.ribbons[key].dispose();
        delete this.ribbons[key];
      }
    }
    for (const t of Object.values(this.trails)) t.color.set(tone(PARTS[Object.keys(this.trails).find((k) => this.trails[k] === t)].color));
    if (!v.echoEvery) this.clearEchoes();
    if (!v.footprints) this.clearFootprints();
    this.marks.visible = v.activity;
  }

  figureColor() {
    if (this.view.figure === 'drawn') return this.stage.isDark() ? '#f4efe6' : '#2f2c28';
    if (this.view.figure === 'body') return this.stage.isDark() ? '#bdb3a6' : '#c9bfb2';
    return this.color;
  }

  setColor(c) {
    this.color = c;
    this.figure?.setColor(this.figureColor());
  }

  clear() {
    for (const t of Object.values(this.trails)) t.clear();
    for (const r of Object.values(this.ribbons)) r.clear();
    this.clearEchoes();
    this.clearFootprints();
  }

  clearEchoes() {
    for (const e of this.echoes) e.fig.dispose();
    this.echoes = [];
  }

  clearFootprints() {
    for (const f of this.footprints) this.stage.scene.remove(f.mesh);
    this.footprints = [];
  }

  /**
   * One played frame.
   * info: { activity?: { joints: number[], planted: [bool, bool] } }
   */
  update(local, info = {}) {
    const sk = this.sk;
    this.lastLocal = local;
    this.figure?.setPose(local);
    sk.world(local, this.world);
    const w = this.world;
    const P = (j) => [w[j * TF], w[j * TF + 1], w[j * TF + 2]];
    for (const [key, trail] of Object.entries(this.trails)) {
      const p = P(sk.jointIndex(PARTS[key].joint));
      trail.push(p[0], key === 'hips' ? 0.005 : p[1], p[2]); // the pelvis trail is drawn on the floor
    }
    for (const [key, ribbon] of Object.entries(this.ribbons)) {
      const [a, b] = RIBBONS[key].map((n) => P(sk.jointIndex(n)));
      ribbon.push(a, b);
    }
    // Echoes: freeze the figure every few seconds.
    const v = this.view;
    if (v.echoEvery > 0) {
      this.sinceEcho += 1 / FPS;
      if (this.sinceEcho >= v.echoEvery) {
        this.sinceEcho = 0;
        this.addEcho(local);
      }
    }
    // Footprints where a foot gets planted.
    const planted = info.activity?.planted;
    if (v.footprints && planted) {
      planted.forEach((on, i) => {
        if (on && !this.prevPlanted?.[i]) this.addFootprint(P(sk.jointIndex(i ? 'RightFoot' : 'LeftFoot')), P(sk.jointIndex(i ? 'RightToeBase' : 'LeftToeBase')), i);
      });
      this.prevPlanted = planted;
    }
    // Corrected joints.
    if (v.activity) {
      const joints = info.activity?.joints || [];
      while (this.marks.children.length < joints.length) this.marks.add(new THREE.Mesh(this.markGeom, this.markMat));
      this.marks.children.forEach((m, i) => {
        m.visible = i < joints.length;
        if (m.visible) m.position.set(...P(joints[i]));
      });
    }
  }

  async addEcho(local) {
    const style = this.view.echoFigure === 'same' ? this.view.figure : this.view.echoFigure;
    const fig = await makeFigure(style === 'body' ? 'solid' : style, this.stage, this.sk, { color: this.figureColor(), opacity: 0.5 });
    fig.setPose(local);
    this.echoes.push({ fig, age: 0 });
    while (this.echoes.length > this.view.echoCount) this.echoes.shift().fig.dispose();
  }

  addFootprint(ankle, toe, side) {
    const mesh = new THREE.Mesh(this.footGeom, new THREE.MeshBasicMaterial({ color: side ? PARTS.footR.color : PARTS.footL.color, transparent: true, opacity: 0.55, depthWrite: false }));
    mesh.position.set((ankle[0] + toe[0]) / 2, 0.003, (ankle[2] + toe[2]) / 2);
    mesh.rotation.y = Math.atan2(toe[0] - ankle[0], toe[2] - ankle[2]);
    this.stage.scene.add(mesh);
    this.footprints.push({ mesh, age: 0 });
    while (this.footprints.length > 80) this.stage.scene.remove(this.footprints.shift().mesh);
  }

  /** Per render frame: fade echoes and footprints, redraw trails. */
  render(dt) {
    const bg = this.stage.scene.background;
    for (const t of Object.values(this.trails)) t.render(bg);
    for (const r of Object.values(this.ribbons)) r.render(bg);
    const n = this.echoes.length;
    this.echoes.forEach((e, i) => e.fig.setOpacity(0.06 + 0.4 * ((i + 1) / n) ** 2));
    for (const f of this.footprints) {
      f.age += dt;
      f.mesh.material.opacity = Math.max(0, 0.55 * (1 - f.age / 20));
    }
  }
}
