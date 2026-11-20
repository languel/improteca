// Quaternion and rigid-transform math, matching NVIDIA Warp's conventions,
// because MotionBricks' runtime is written in Warp and the port must agree
// with it numerically.
//
// Quaternions are [x, y, z, w]. A transform is 7 numbers: position xyz, then
// rotation xyzw, stored in flat arrays (one pose = joints * 7 values).

export const TF = 7;
const EPS = 1e-8;

export function quat(x = 0, y = 0, z = 0, w = 1) {
  return [x, y, z, w];
}

export function qmul(a, b) {
  return [
    a[3] * b[0] + b[3] * a[0] + a[1] * b[2] - b[1] * a[2],
    a[3] * b[1] + b[3] * a[1] + a[2] * b[0] - b[2] * a[0],
    a[3] * b[2] + b[3] * a[2] + a[0] * b[1] - b[0] * a[1],
    a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  ];
}

export function qinv(q) {
  return [-q[0], -q[1], -q[2], q[3]];
}

export function qnormalize(q) {
  const l = Math.hypot(q[0], q[1], q[2], q[3]);
  if (l > EPS) return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
  return [0, 0, 0, 1];
}

/** Warp's quat_rotate. */
export function qrot(q, v) {
  const c = 2 * q[3] * q[3] - 1;
  const d = 2 * (q[0] * v[0] + q[1] * v[1] + q[2] * v[2]);
  return [
    v[0] * c + q[0] * d + (q[1] * v[2] - q[2] * v[1]) * q[3] * 2,
    v[1] * c + q[1] * d + (q[2] * v[0] - q[0] * v[2]) * q[3] * 2,
    v[2] * c + q[2] * d + (q[0] * v[1] - q[1] * v[0]) * q[3] * 2,
  ];
}

/** Warp's quat_rotate_inv. */
export function qrotInv(q, v) {
  const c = 2 * q[3] * q[3] - 1;
  const d = 2 * (q[0] * v[0] + q[1] * v[1] + q[2] * v[2]);
  return [
    v[0] * c + q[0] * d - (q[1] * v[2] - q[2] * v[1]) * q[3] * 2,
    v[1] * c + q[1] * d - (q[2] * v[0] - q[0] * v[2]) * q[3] * 2,
    v[2] * c + q[2] * d - (q[0] * v[1] - q[1] * v[0]) * q[3] * 2,
  ];
}

export function qFromAxisAngle(axis, angle) {
  const s = Math.sin(angle * 0.5);
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle * 0.5)];
}

function vnormalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > EPS ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
}

/** Warp's quat_to_axis_angle: the shortest rotation (angle in [0, pi]). */
export function qToAxisAngle(q) {
  const v = [q[0], q[1], q[2]];
  let axis = vnormalize(v);
  if (q[3] < 0) axis = [-axis[0], -axis[1], -axis[2]];
  const angle = 2 * Math.atan2(Math.hypot(v[0], v[1], v[2]), Math.abs(q[3]));
  return { axis, angle };
}

/** Warp's quat_slerp (via the relative rotation's axis and angle). */
export function qslerp(q0, q1, t) {
  const { axis, angle } = qToAxisAngle(qmul(qinv(q0), q1));
  return qmul(q0, qFromAxisAngle(axis, t * angle));
}

/**
 * Warp's quat_from_matrix for a 3x3 matrix given as columns (x, y, z axes),
 * i.e. wp.quat_from_matrix(wp.matrix_from_cols(x, y, z)).
 */
export function qFromCols(x, y, z) {
  // m[row][col]
  const m = [
    [x[0], y[0], z[0]],
    [x[1], y[1], z[1]],
    [x[2], y[2], z[2]],
  ];
  const tr = m[0][0] + m[1][1] + m[2][2];
  let qx, qy, qz, qw, h;
  if (tr >= 0) {
    h = Math.sqrt(tr + 1);
    qw = 0.5 * h;
    h = 0.5 / h;
    qx = (m[2][1] - m[1][2]) * h;
    qy = (m[0][2] - m[2][0]) * h;
    qz = (m[1][0] - m[0][1]) * h;
  } else {
    let maxDiag = 0;
    if (m[1][1] > m[0][0]) maxDiag = 1;
    if (m[2][2] > m[maxDiag][maxDiag]) maxDiag = 2;
    if (maxDiag === 0) {
      h = Math.sqrt(m[0][0] - (m[1][1] + m[2][2]) + 1);
      qx = 0.5 * h;
      h = 0.5 / h;
      qy = (m[0][1] + m[1][0]) * h;
      qz = (m[2][0] + m[0][2]) * h;
      qw = (m[2][1] - m[1][2]) * h;
    } else if (maxDiag === 1) {
      h = Math.sqrt(m[1][1] - (m[2][2] + m[0][0]) + 1);
      qy = 0.5 * h;
      h = 0.5 / h;
      qz = (m[1][2] + m[2][1]) * h;
      qx = (m[0][1] + m[1][0]) * h;
      qw = (m[0][2] - m[2][0]) * h;
    } else {
      h = Math.sqrt(m[2][2] - (m[0][0] + m[1][1]) + 1);
      qz = 0.5 * h;
      h = 0.5 / h;
      qx = (m[2][0] + m[0][2]) * h;
      qy = (m[1][2] + m[2][1]) * h;
      qw = (m[1][0] - m[0][1]) * h;
    }
  }
  return qnormalize([qx, qy, qz, qw]);
}

export function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export { vnormalize };

// ---- transforms ------------------------------------------------------------

export function tf(p = [0, 0, 0], q = [0, 0, 0, 1]) {
  return [p[0], p[1], p[2], q[0], q[1], q[2], q[3]];
}

export const IDENTITY = Object.freeze([0, 0, 0, 0, 0, 0, 1]);

export function tp(t) {
  return [t[0], t[1], t[2]];
}

export function tq(t) {
  return [t[3], t[4], t[5], t[6]];
}

/** Read transform i from a flat array. */
export function getTf(arr, i) {
  const o = i * TF;
  return [arr[o], arr[o + 1], arr[o + 2], arr[o + 3], arr[o + 4], arr[o + 5], arr[o + 6]];
}

export function setTf(arr, i, t) {
  const o = i * TF;
  for (let k = 0; k < TF; k++) arr[o + k] = t[k];
}

/** wp.mul(a, b) for transforms: apply b in a's frame. */
export function tmul(a, b) {
  const aq = tq(a);
  const p = qrot(aq, tp(b));
  const q = qmul(aq, tq(b));
  return [a[0] + p[0], a[1] + p[1], a[2] + p[2], q[0], q[1], q[2], q[3]];
}

export function tinv(t) {
  const qi = qinv(tq(t));
  const p = qrot(qi, tp(t));
  return [-p[0], -p[1], -p[2], qi[0], qi[1], qi[2], qi[3]];
}

/** Linear position, spherical rotation. */
export function tblend(a, b, s) {
  const q = qslerp(tq(a), tq(b), s);
  return [
    a[0] + (b[0] - a[0]) * s,
    a[1] + (b[1] - a[1]) * s,
    a[2] + (b[2] - a[2]) * s,
    q[0], q[1], q[2], q[3],
  ];
}

/** pose_utils.compute_delta_transform */
export function deltaTransform(src, dst) {
  return tmul(tinv(src), dst);
}

/** pose_utils.apply_delta_transform */
export function applyDelta(t, delta, weight) {
  return tmul(t, tblend(IDENTITY, delta, weight));
}

/** pose_utils.compute_root_endpoint_delta */
export function rootEndpointDelta(generated, guide) {
  const r = qnormalize(qmul(qinv(qnormalize(tq(generated))), qnormalize(tq(guide))));
  return [guide[0] - generated[0], guide[1] - generated[1], guide[2] - generated[2], ...r];
}

/** pose_utils.apply_root_endpoint_correction */
export function applyRootCorrection(root, startDelta, endDelta, endWeight) {
  const startWeight = 1 - endWeight;
  const id = [0, 0, 0, 1];
  const q = qnormalize(
    qmul(qmul(tq(root), qslerp(id, tq(startDelta), startWeight)), qslerp(id, tq(endDelta), endWeight)),
  );
  return [
    root[0] + startDelta[0] * startWeight + endDelta[0] * endWeight,
    root[1] + startDelta[1] * startWeight + endDelta[1] * endWeight,
    root[2] + startDelta[2] * startWeight + endDelta[2] * endWeight,
    ...q,
  ];
}

/** A yaw rotation about +Y and a ground position, the usual way to place a pose. */
export function placement(x, z, yawRadians) {
  return tf([x, 0, z], qFromAxisAngle([0, 1, 0], yawRadians));
}
