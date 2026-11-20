// Figures: ways to draw a SOMA pose. All share one interface:
//   new Figure(stage, skeleton, options); fig.setPose(local); fig.setColor(c);
//   fig.setOpacity(a); fig.visible = bool; fig.dispose(); fig.root() -> [x,y,z]
//
//   sticks     bones as thin cylinders, a round head (functional, light)
//   solid      a procedural mannequin of tapered limbs, torso and head
//   body       NVIDIA's SOMA body mesh, skinned to the skeleton (built by tools/build_body.py)
//   drawn      loose, slightly offset strokes per bone, like a drawing (after Ghostcatching)
//   points     joints only, like motion-capture markers

import * as THREE from 'three';
import { TF } from '../src/engine/math.js';

const FINGER = /Hand(Thumb|Index|Middle|Ring|Pinky)/;
const SKIP = /(End|Eye|Jaw)$/;
const up = new THREE.Vector3(0, 1, 0);

function bones(skeleton, { fingers = true } = {}) {
  const out = [];
  for (let j = 1; j < skeleton.numJoints; j++) {
    const name = skeleton.names[j];
    if (SKIP.test(name) && !/HeadEnd/.test(name)) continue;
    if (!fingers && FINGER.test(name)) continue;
    out.push({ j, p: skeleton.parents[j], name, parentName: skeleton.names[skeleton.parents[j]] });
  }
  return out;
}

class Base {
  constructor(stage, skeleton) {
    this.stage = stage;
    this.skeleton = skeleton;
    this.group = new THREE.Group();
    this.world = new Float64Array(skeleton.numJoints * TF);
    stage.scene.add(this.group);
  }
  set visible(v) {
    this.group.visible = v;
  }
  get visible() {
    return this.group.visible;
  }
  pos(j, v = new THREE.Vector3()) {
    const w = this.world;
    return v.set(w[j * TF], w[j * TF + 1], w[j * TF + 2]);
  }
  root() {
    return [this.world[0], this.world[1], this.world[2]];
  }
  dispose() {
    this.stage.scene.remove(this.group);
    this.group.traverse((o) => {
      o.geometry?.dispose?.();
      if (o.material && !o.material.shared) o.material.dispose?.();
    });
  }
}

/** Bones as cylinders between joints. Radius per bone decides sticks vs solid. */
class Limbs extends Base {
  constructor(stage, skeleton, { color = '#2f2c28', opacity = 1, solid = false } = {}) {
    super(stage, skeleton);
    const transparent = opacity < 1;
    this.material = new THREE.MeshStandardMaterial({ color, roughness: solid ? 0.8 : 0.55, metalness: 0, transparent, opacity, depthWrite: !transparent });
    this.solid = solid;
    const unit = new THREE.CylinderGeometry(1, 1, 1, solid ? 16 : 10, 1);
    unit.translate(0, 0.5, 0);
    const taper = new THREE.CylinderGeometry(0.75, 1, 1, 16, 1); // narrower at the far joint
    taper.translate(0, 0.5, 0);
    this.items = [];
    for (const b of bones(skeleton, { fingers: !solid })) {
      const finger = FINGER.test(b.name);
      let r;
      if (!solid) r = finger ? 0.006 : /Shin|Leg|Spine|Chest|Hips/.test(b.parentName) ? 0.05 : 0.035;
      else r = SOLID_RADIUS[b.parentName] ?? (finger ? 0.008 : 0.03);
      const mesh = new THREE.Mesh(solid && /Leg|Shin|Arm|ForeArm/.test(b.parentName) ? taper : unit, this.material);
      mesh.castShadow = !transparent;
      mesh.userData = { ...b, r };
      this.group.add(mesh);
      this.items.push(mesh);
    }
    if (solid) {
      // Joints as spheres so limbs read as one surface; torso and pelvis as ellipsoids.
      const ball = new THREE.SphereGeometry(1, 16, 12);
      this.balls = [];
      for (const [name, r] of Object.entries(SOLID_JOINTS)) {
        const m = new THREE.Mesh(ball, this.material);
        m.castShadow = !transparent;
        m.userData = { j: skeleton.jointIndex(name), r };
        this.group.add(m);
        this.balls.push(m);
      }
      this.chest = new THREE.Mesh(ball, this.material);
      this.pelvis = new THREE.Mesh(ball, this.material);
      for (const m of [this.chest, this.pelvis]) {
        m.castShadow = !transparent;
        this.group.add(m);
      }
    }
    this.head = new THREE.Mesh(new THREE.SphereGeometry(solid ? 0.1 : 0.105, 24, 16), this.material);
    if (solid) this.head.scale.set(0.85, 1.1, 0.95);
    this.head.castShadow = !transparent;
    this.group.add(this.head);
    this.idx = Object.fromEntries(['Head', 'HeadEnd', 'Chest', 'Spine2', 'Hips', 'LeftArm', 'RightArm', 'LeftLeg', 'RightLeg', 'Neck1'].map((n) => [n, skeleton.jointIndex(n)]));
  }
  setColor(c) {
    this.material.color.set(c);
  }
  setOpacity(a) {
    this.material.opacity = a;
    this.material.transparent = a < 1;
    this.material.depthWrite = a >= 1;
  }
  setPose(local) {
    this.skeleton.world(local, this.world);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    for (const mesh of this.items) {
      const { j, p, r } = mesh.userData;
      this.pos(p, a);
      this.pos(j, b);
      const d = b.clone().sub(a);
      const len = d.length();
      mesh.position.copy(a);
      mesh.scale.set(r, Math.max(len, 1e-4), r);
      if (len > 1e-6) mesh.quaternion.setFromUnitVectors(up, d.divideScalar(len));
    }
    const I = this.idx;
    const h = this.pos(I.Head);
    const e = this.pos(I.HeadEnd);
    this.head.position.copy(h).add(e).multiplyScalar(0.5);
    if (this.solid) {
      this.head.quaternion.setFromUnitVectors(up, e.clone().sub(h).normalize());
      for (const m of this.balls) {
        this.pos(m.userData.j, m.position);
        m.scale.setScalar(m.userData.r);
      }
      // Chest: between the shoulders and the spine; pelvis: between the hips.
      const ls = this.pos(I.LeftArm);
      const rs = this.pos(I.RightArm);
      const sp = this.pos(I.Spine2);
      const neck = this.pos(I.Neck1);
      this.chest.position.copy(ls).add(rs).multiplyScalar(0.5).lerp(sp, 0.45);
      const across = ls.clone().sub(rs);
      const upward = neck.clone().sub(sp);
      orient(this.chest, across, upward, [across.length() * 0.42, upward.length() * 0.62, 0.11]);
      const ll = this.pos(I.LeftLeg);
      const rl = this.pos(I.RightLeg);
      const hips = this.pos(I.Hips);
      this.pelvis.position.copy(ll).add(rl).multiplyScalar(0.5).lerp(hips, 0.5);
      orient(this.pelvis, ll.clone().sub(rl), hips.clone().sub(ll.clone().add(rl).multiplyScalar(0.5)).add(sp.clone().sub(hips)), [0.17, 0.1, 0.11]);
    }
  }
}

/** Point an ellipsoid with axes along `across` and `upward`, with radii [x, y, z]. */
function orient(mesh, across, upward, radii) {
  const x = across.clone().normalize();
  const y = upward.clone().sub(x.clone().multiplyScalar(upward.dot(x))).normalize();
  const z = new THREE.Vector3().crossVectors(x, y);
  mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
  mesh.scale.set(...radii);
}

const SOLID_RADIUS = {
  LeftLeg: 0.075, RightLeg: 0.075, LeftShin: 0.055, RightShin: 0.055, LeftFoot: 0.035, RightFoot: 0.035,
  LeftArm: 0.045, RightArm: 0.045, LeftForeArm: 0.035, RightForeArm: 0.035, LeftHand: 0.025, RightHand: 0.025,
  Hips: 0.08, Spine1: 0.09, Spine2: 0.1, Chest: 0.06, Neck1: 0.045, Neck2: 0.045, Head: 0.04,
  LeftShoulder: 0.05, RightShoulder: 0.05, LeftToeBase: 0.025, RightToeBase: 0.025,
};
const SOLID_JOINTS = {
  LeftArm: 0.05, RightArm: 0.05, LeftForeArm: 0.04, RightForeArm: 0.04, LeftHand: 0.035, RightHand: 0.035,
  LeftLeg: 0.08, RightLeg: 0.08, LeftShin: 0.06, RightShin: 0.06, LeftFoot: 0.045, RightFoot: 0.045,
  Neck1: 0.05,
};

/** Loose strokes: each bone drawn as a few slightly offset lines, like a sketch. */
class Drawn extends Base {
  constructor(stage, skeleton, { color = '#f4efe6', opacity = 0.9, strokes = 3, seed = 1 } = {}) {
    super(stage, skeleton);
    this.list = bones(skeleton, { fingers: false });
    this.strokes = strokes;
    let s = seed >>> 0;
    const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296 - 0.5);
    // Per bone and stroke: offsets at both ends and a slight overshoot, fixed for this figure.
    this.jitter = this.list.flatMap(() => Array.from({ length: strokes }, () => [rnd() * 0.03, rnd() * 0.03, rnd() * 0.03, rnd() * 0.03, rnd() * 0.03, rnd() * 0.03, 0.04 + rnd() * 0.06]));
    const count = this.list.length * strokes * 2;
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3));
    this.material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false });
    this.lines = new THREE.LineSegments(this.geometry, this.material);
    this.lines.frustumCulled = false;
    this.group.add(this.lines);
    // Head as a loose ring of strokes.
    this.headGeom = new THREE.BufferGeometry();
    const ring = [];
    for (let k = 0; k < 2; k++) for (let i = 0; i <= 20; i++) ring.push(i / 20, k);
    this.ringT = ring;
    this.headGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array((ring.length / 2) * 3), 3));
    this.headLine = new THREE.Line(this.headGeom, this.material);
    this.headLine.frustumCulled = false;
    this.group.add(this.headLine);
    this.hi = skeleton.jointIndex('Head');
    this.he = skeleton.jointIndex('HeadEnd');
  }
  setColor(c) {
    this.material.color.set(c);
  }
  setOpacity(a) {
    this.material.opacity = a;
  }
  setPose(local) {
    this.skeleton.world(local, this.world);
    const arr = this.geometry.attributes.position.array;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    let o = 0;
    this.list.forEach((bone, i) => {
      this.pos(bone.p, a);
      this.pos(bone.j, b);
      const d = b.clone().sub(a);
      for (let k = 0; k < this.strokes; k++) {
        const J = this.jitter[i * this.strokes + k];
        const over = J[6];
        arr[o++] = a.x - d.x * over + J[0];
        arr[o++] = a.y - d.y * over + J[1];
        arr[o++] = a.z - d.z * over + J[2];
        arr[o++] = b.x + d.x * over + J[3];
        arr[o++] = b.y + d.y * over + J[4];
        arr[o++] = b.z + d.z * over + J[5];
      }
    });
    this.geometry.attributes.position.needsUpdate = true;
    // Head: two loose circles around the head's centre, facing the camera roughly.
    const h = this.pos(this.hi);
    const e = this.pos(this.he);
    const c = h.clone().add(e).multiplyScalar(0.5);
    const axis = e.clone().sub(h).normalize();
    const side = new THREE.Vector3(1, 0, 0).sub(axis.clone().multiplyScalar(axis.x)).normalize();
    const fwd = new THREE.Vector3().crossVectors(axis, side);
    const hp = this.headGeom.attributes.position.array;
    for (let i = 0; i < this.ringT.length / 2; i++) {
      const t = this.ringT[i * 2] * Math.PI * 2;
      const k = this.ringT[i * 2 + 1];
      const r = 0.1 + k * 0.012;
      const p = c.clone().add(axis.clone().multiplyScalar(Math.cos(t) * r * 1.15)).add((k ? fwd : side).clone().multiplyScalar(Math.sin(t) * r));
      hp[i * 3] = p.x;
      hp[i * 3 + 1] = p.y;
      hp[i * 3 + 2] = p.z;
    }
    this.headGeom.attributes.position.needsUpdate = true;
  }
}

/** Joints as small dots, like motion-capture markers. */
class Points extends Base {
  constructor(stage, skeleton, { color = '#2f2c28', opacity = 1 } = {}) {
    super(stage, skeleton);
    this.list = [0, ...bones(skeleton, { fingers: false }).map((b) => b.j)];
    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(this.list.length * 3), 3));
    this.material = new THREE.PointsMaterial({ color, size: 0.045, transparent: opacity < 1, opacity });
    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.group.add(this.points);
  }
  setColor(c) {
    this.material.color.set(c);
  }
  setOpacity(a) {
    this.material.opacity = a;
    this.material.transparent = a < 1;
  }
  setPose(local) {
    this.skeleton.world(local, this.world);
    const arr = this.geometry.attributes.position.array;
    this.list.forEach((j, i) => {
      arr[i * 3] = this.world[j * TF];
      arr[i * 3 + 1] = this.world[j * TF + 1];
      arr[i * 3 + 2] = this.world[j * TF + 2];
    });
    this.geometry.attributes.position.needsUpdate = true;
  }
}

/** NVIDIA's SOMA body, skinned. Load the data once with loadBody(). */
class Body extends Base {
  constructor(stage, skeleton, { color = '#c9bfb2', opacity = 1, data }) {
    super(stage, skeleton);
    const J = skeleton.numJoints;
    this.bones = [];
    for (let j = 0; j < J; j++) {
      const bone = new THREE.Bone();
      bone.name = skeleton.names[j];
      this.bones.push(bone);
      if (j > 0) this.bones[skeleton.parents[j]].add(bone);
    }
    // Bone inverses: the USD bind pose where given, else the rest pose.
    const rest = skeleton.world(skeleton.reference);
    const inverses = this.bones.map((_, j) => {
      const m = new THREE.Matrix4();
      if (data.bind[j]) m.fromArray(data.bind[j]);
      else m.compose(new THREE.Vector3(rest[j * TF], rest[j * TF + 1], rest[j * TF + 2]), new THREE.Quaternion(rest[j * TF + 3], rest[j * TF + 4], rest[j * TF + 5], rest[j * TF + 6]), new THREE.Vector3(1, 1, 1));
      return m.invert();
    });
    const transparent = opacity < 1;
    this.material = new THREE.MeshStandardMaterial({ color, roughness: 0.7, metalness: 0, transparent, opacity, depthWrite: !transparent });
    this.mesh = new THREE.SkinnedMesh(data.geometry, this.material);
    this.mesh.castShadow = !transparent;
    this.mesh.frustumCulled = false;
    this.mesh.add(this.bones[0]);
    this.mesh.bind(new THREE.Skeleton(this.bones, inverses), new THREE.Matrix4());
    this.group.add(this.mesh);
  }
  setColor(c) {
    this.material.color.set(c);
  }
  setOpacity(a) {
    this.material.opacity = a;
    this.material.transparent = a < 1;
    this.material.depthWrite = a >= 1;
  }
  setPose(local) {
    this.skeleton.world(local, this.world);
    for (let j = 0; j < this.bones.length; j++) {
      const o = j * TF;
      this.bones[j].position.set(local[o], local[o + 1], local[o + 2]);
      this.bones[j].quaternion.set(local[o + 3], local[o + 4], local[o + 5], local[o + 6]);
    }
  }
  dispose() {
    this.stage.scene.remove(this.group);
    this.material.dispose();
  }
}

let bodyPromise = null;
/** Load the SOMA body built by tools/build_body.py (served under models/body/). */
export function loadBody() {
  bodyPromise ??= (async () => {
    const base = new URL('../models/body/', import.meta.url);
    const meta = await (await fetch(new URL('soma_body.json', base))).json();
    const buf = await (await fetch(new URL('soma_body.bin', base))).arrayBuffer();
    const view = (name, Type) => new Type(buf, meta.layout[name].offset, meta.layout[name].length);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(view('position', Float32Array), 3));
    g.setIndex(new THREE.BufferAttribute(view('index', Uint32Array), 1));
    g.setAttribute('skinIndex', new THREE.BufferAttribute(view('skinIndex', Uint16Array), 4));
    g.setAttribute('skinWeight', new THREE.BufferAttribute(view('skinWeight', Float32Array), 4));
    g.computeVertexNormals();
    return { geometry: g, bind: meta.bind };
  })();
  bodyPromise.catch(() => (bodyPromise = null));
  return bodyPromise;
}

export const FIGURES = {
  sticks: { label: 'Sticks', make: (stage, sk, o) => new Limbs(stage, sk, o) },
  solid: { label: 'Mannequin', make: (stage, sk, o) => new Limbs(stage, sk, { ...o, solid: true }) },
  body: { label: 'Body (SOMA)', make: (stage, sk, o) => new Body(stage, sk, o), needsBody: true },
  drawn: { label: 'Drawn lines', make: (stage, sk, o) => new Drawn(stage, sk, o) },
  points: { label: 'Markers', make: (stage, sk, o) => new Points(stage, sk, o) },
};

/** Make a figure of a style; the body style needs loadBody() first (falls back to the mannequin). */
export async function makeFigure(style, stage, skeleton, options = {}) {
  const def = FIGURES[style] || FIGURES.sticks;
  if (def.needsBody) {
    try {
      return def.make(stage, skeleton, { ...options, data: await loadBody() });
    } catch (err) {
      console.warn('SOMA body unavailable; using the mannequin', err);
      return FIGURES.solid.make(stage, skeleton, options);
    }
  }
  return def.make(stage, skeleton, options);
}
