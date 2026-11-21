// The stage: a three.js scene with a floor and simple mannequins drawn from
// SOMA poses (local joint transforms, Y up, meters).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TF } from '../src/engine/math.js';

const FINGER = /Hand(Thumb|Index|Middle|Ring|Pinky)/;
const SKIP = /(End|Eye|Jaw)$/;

export class Stage {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.05, 200);
    this.camera.position.set(5.5, 3.2, 8);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.9, 0);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 1.6));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(4, 8, 5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8 });
    this.scene.add(sun);

    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ roughness: 0.95 }));
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.receiveShadow = true;
    this.floor.userData = { prop: 'floor' };
    this.scene.add(this.floor);
    this.grid = new THREE.GridHelper(40, 40);
    this.grid.position.y = 0.001;
    this.scene.add(this.grid);
    this.trail = null;
    this.buildEnvironmentProps();
    this.setupPicking();

    this.bgMode = 'auto';
    this.applyTheme();
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => this.applyTheme());
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  /** Build interactive 3D stage props (Table, Chair, Glass, and Point-to-Go Beacon). */
  buildEnvironmentProps() {
    this.propsGroup = new THREE.Group();
    this.scene.add(this.propsGroup);
    this.propPickTargets = [];

    const woodMat = new THREE.MeshStandardMaterial({ color: '#8b6848', roughness: 0.72, metalness: 0.05 });
    const darkWoodMat = new THREE.MeshStandardMaterial({ color: '#6b4f35', roughness: 0.75, metalness: 0.05 });
    const tagAll = (group, prop) => {
      group.userData.prop = prop;
      group.traverse((child) => {
        if (child.isMesh) {
          child.castShadow = true;
          child.receiveShadow = true;
          child.userData.prop = prop;
          this.propPickTargets.push(child);
        }
      });
    };

    // Table at (1.35, 0, -0.78), top height 0.72 m, radius 0.42 m.
    const table = new THREE.Group();
    table.position.set(1.35, 0, -0.78);
    const top = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.42, 0.036, 28), woodMat);
    top.position.y = 0.702;
    table.add(top);
    const legGeom = new THREE.CylinderGeometry(0.022, 0.018, 0.69, 12);
    for (const [lx, lz] of [[0.26, 0.26], [-0.26, 0.26], [0.26, -0.26], [-0.26, -0.26]]) {
      const leg = new THREE.Mesh(legGeom, darkWoodMat);
      leg.position.set(lx, 0.345, lz);
      table.add(leg);
    }
    tagAll(table, 'table');
    this.propsGroup.add(table);
    this.tableGroup = table;

    // Chair at (1.35, 0, -1.45), facing +Z toward the table, seat height 0.48 m.
    const chair = new THREE.Group();
    chair.position.set(1.35, 0, -1.45);
    const seat = new THREE.Mesh(new THREE.BoxGeometry(0.44, 0.035, 0.42), woodMat);
    seat.position.y = 0.465;
    chair.add(seat);
    const cLegGeom = new THREE.CylinderGeometry(0.018, 0.014, 0.45, 10);
    for (const [lx, lz] of [[0.18, 0.17], [-0.18, 0.17], [0.18, -0.17], [-0.18, -0.17]]) {
      const leg = new THREE.Mesh(cLegGeom, darkWoodMat);
      leg.position.set(lx, 0.225, lz);
      chair.add(leg);
    }
    const backPostGeom = new THREE.CylinderGeometry(0.016, 0.016, 0.42, 10);
    for (const lx of [-0.18, 0.18]) {
      const post = new THREE.Mesh(backPostGeom, darkWoodMat);
      post.position.set(lx, 0.68, -0.18);
      chair.add(post);
    }
    const backrest = new THREE.Mesh(new THREE.BoxGeometry(0.42, 0.16, 0.028), woodMat);
    backrest.position.set(0, 0.78, -0.18);
    chair.add(backrest);
    tagAll(chair, 'chair');
    this.propsGroup.add(chair);
    this.chairGroup = chair;

    // Glass on the table at (1.24, 0.76, -0.98).
    const glass = new THREE.Group();
    glass.position.set(1.24, 0.76, -0.98);
    const cupMat = new THREE.MeshStandardMaterial({ color: '#d8eef5', roughness: 0.18, metalness: 0.1, transparent: true, opacity: 0.72 });
    const liquidMat = new THREE.MeshStandardMaterial({ color: '#2b8cbe', roughness: 0.25, metalness: 0.05, transparent: true, opacity: 0.88 });
    const cup = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.030, 0.11, 18), cupMat);
    cup.position.y = 0.015;
    glass.add(cup);
    const liquid = new THREE.Mesh(new THREE.CylinderGeometry(0.033, 0.027, 0.072, 16), liquidMat);
    liquid.position.y = 0.005;
    glass.add(liquid);
    tagAll(glass, 'glass');
    this.propsGroup.add(glass);
    this.glassGroup = glass;

    // Floor beacon ring for Point-to-Go gesture / click steering.
    const beaconGeom = new THREE.RingGeometry(0.11, 0.16, 32).rotateX(-Math.PI / 2);
    const beaconMat = new THREE.MeshBasicMaterial({ color: '#00798c', transparent: true, opacity: 0.75, side: THREE.DoubleSide });
    this.beaconMesh = new THREE.Mesh(beaconGeom, beaconMat);
    this.beaconMesh.position.set(0, 0.006, 0);
    this.beaconMesh.visible = false;
    this.scene.add(this.beaconMesh);
  }

  /** Sync 3D glass position/tilt and floor beacon with the current environment state. */
  updateEnvironment(env) {
    if (!env || !this.glassGroup) return;
    if (env.glass) {
      this.glassGroup.position.set(env.glass.x, env.glass.y, env.glass.z);
      this.glassGroup.rotation.set(env.glass.tilt || 0, 0, 0);
    }
    if (env.beacon && this.beaconMesh) {
      this.beaconMesh.visible = Boolean(env.beacon.active);
      if (env.beacon.active) {
        this.beaconMesh.position.set(env.beacon.x, 0.006, env.beacon.z);
      }
    }
  }

  /** Enable click/tap raycasting on props (table, chair, glass) and floor (point-to-go). */
  setupPicking() {
    this.onPick = null;
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let down = null;
    const dom = this.renderer.domElement;
    dom.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      down = { x: e.clientX, y: e.clientY, time: performance.now() };
    });
    dom.addEventListener('pointerup', (e) => {
      if (!down || !this.onPick) return;
      const dist = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      const dt = performance.now() - down.time;
      down = null;
      if (dist > 7 || dt > 600) return; // camera orbit drag, not a click
      const rect = dom.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, this.camera);
      // Check props first (glass, chair, table), then the stage floor.
      const propHits = raycaster.intersectObjects(this.propPickTargets, false);
      if (propHits.length) {
        const prop = propHits[0].object.userData.prop;
        this.onPick({ kind: 'prop', prop, point: propHits[0].point, shift: e.shiftKey });
        return;
      }
      const floorHits = raycaster.intersectObject(this.floor, false);
      if (floorHits.length) {
        const pt = floorHits[0].point;
        if (Math.hypot(pt.x, pt.z) <= 5.5) {
          this.onPick({ kind: 'floor', x: +pt.x.toFixed(2), z: +pt.z.toFixed(2), shift: e.shiftKey });
        }
      }
    });
  }

  /** 'auto' follows the page's theme; 'light' and 'dark' override it (e.g. for a drawn look). */
  setBackground(mode = 'auto') {
    this.bgMode = mode;
    this.applyTheme();
  }

  isDark() {
    if (this.bgMode === 'dark') return true;
    if (this.bgMode === 'light') return false;
    return matchMedia('(prefers-color-scheme: dark)').matches && document.documentElement.dataset.theme !== 'light';
  }

  applyTheme() {
    const css = getComputedStyle(document.documentElement);
    const get = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
    let bg = get('--stage-bg', '#ece9e2');
    let floor = get('--stage-floor', '#e2ded5');
    let grid = get('--stage-grid', '#cfc9bd');
    if (this.bgMode === 'dark') [bg, floor, grid] = ['#0d0c0b', '#121110', '#1f1d1a'];
    if (this.bgMode === 'light') [bg, floor, grid] = ['#ece9e2', '#e2ded5', '#cfc9bd'];
    this.scene.background = new THREE.Color(bg);
    this.floor.material.color.set(floor);
    this.grid.material.color = new THREE.Color(grid);
    this.grid.material.vertexColors = false;
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  /** Move the view with a point (e.g. the dancer's root), keeping the camera's offset. */
  follow(point, amount = 0.06) {
    const t = this.controls.target;
    const dx = (point[0] - t.x) * amount;
    const dz = (point[2] - t.z) * amount;
    t.x += dx;
    t.z += dz;
    this.camera.position.x += dx;
    this.camera.position.z += dz;
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  /** Draw a path on the floor (e.g. the root trajectory). */
  setTrail(points, color) {
    if (this.trail) {
      this.scene.remove(this.trail);
      this.trail.geometry.dispose();
    }
    this.trail = null;
    if (!points.length) return;
    const g = new THREE.BufferGeometry().setFromPoints(points.map((p) => new THREE.Vector3(p[0], 0.004, p[2])));
    this.trail = new THREE.Line(g, new THREE.LineBasicMaterial({ color }));
    this.scene.add(this.trail);
  }
}

/** A stick-and-joint mannequin for one skeleton. */
export class Mannequin {
  constructor(stage, skeleton, { color = '#c8553d', opacity = 1, label = '' } = {}) {
    this.stage = stage;
    this.skeleton = skeleton;
    this.group = new THREE.Group();
    const transparent = opacity < 1;
    this.material = new THREE.MeshStandardMaterial({ color, roughness: 0.55, transparent, opacity, depthWrite: !transparent });
    const J = skeleton.numJoints;
    this.bones = [];
    const unit = new THREE.CylinderGeometry(1, 1, 1, 10, 1);
    unit.translate(0, 0.5, 0);
    for (let j = 1; j < J; j++) {
      const name = skeleton.names[j];
      if (SKIP.test(name) && !/HeadEnd/.test(name)) continue;
      const finger = FINGER.test(name);
      const radius = finger ? 0.006 : /Shin|Leg|Spine|Chest|Hips/.test(skeleton.names[skeleton.parents[j]]) ? 0.05 : 0.035;
      const mesh = new THREE.Mesh(unit, this.material);
      mesh.castShadow = !transparent;
      mesh.userData = { j, p: skeleton.parents[j], radius };
      this.group.add(mesh);
      this.bones.push(mesh);
    }
    this.head = new THREE.Mesh(new THREE.SphereGeometry(0.105, 20, 14), this.material);
    this.head.castShadow = !transparent;
    this.group.add(this.head);
    this.headIndex = skeleton.jointIndex('Head');
    this.headEnd = skeleton.jointIndex('HeadEnd');
    this.world = new Float64Array(J * TF);
    this.label = label;
    stage.scene.add(this.group);
  }

  set visible(v) {
    this.group.visible = v;
  }

  setColor(color) {
    this.material.color.set(color);
  }

  /** Pose from local joint transforms (joints * 7). */
  setPose(local) {
    const w = this.skeleton.world(local, this.world);
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0);
    for (const mesh of this.bones) {
      const { j, p, radius } = mesh.userData;
      a.set(w[p * TF], w[p * TF + 1], w[p * TF + 2]);
      b.set(w[j * TF], w[j * TF + 1], w[j * TF + 2]);
      const d = b.clone().sub(a);
      const len = d.length();
      mesh.position.copy(a);
      mesh.scale.set(radius, Math.max(len, 1e-4), radius);
      if (len > 1e-6) mesh.quaternion.setFromUnitVectors(up, d.divideScalar(len));
    }
    const h = this.headIndex * TF;
    const e = this.headEnd * TF;
    this.head.position.set((w[h] + w[e]) / 2, (w[h + 1] + w[e + 1]) / 2, (w[h + 2] + w[e + 2]) / 2);
  }

  root() {
    return [this.world[0], this.world[1], this.world[2]];
  }

  dispose() {
    this.stage.scene.remove(this.group);
    this.material.dispose();
  }
}
