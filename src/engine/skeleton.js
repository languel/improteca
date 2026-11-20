// A joint hierarchy with reference (rest) local transforms.
// Parents always come before their children.

import { TF, IDENTITY, getTf, setTf, tmul } from './math.js';

export class Skeleton {
  /**
   * @param {string[]} names
   * @param {number[]} parents  -1 for the root
   * @param {ArrayLike<number>} referenceLocal  joints * 7
   */
  constructor(names, parents, referenceLocal) {
    if (names.length !== parents.length) throw new Error('names and parents differ in length');
    if (referenceLocal.length !== names.length * TF) throw new Error('reference pose has the wrong size');
    parents.forEach((p, i) => {
      if (p >= i) throw new Error(`joint ${names[i]} comes before its parent`);
    });
    this.names = names;
    this.parents = parents;
    this.reference = Float64Array.from(referenceLocal);
    this.index = new Map(names.map((n, i) => [n, i]));
  }

  get numJoints() {
    return this.names.length;
  }

  jointIndex(name) {
    return this.index.has(name) ? this.index.get(name) : -1;
  }

  /** Local transforms of one pose -> world transforms (pose_utils.compute_world_pose_frame). */
  world(local, out = new Float64Array(local.length), rootTx = IDENTITY, inPose = 0, outPose = inPose) {
    const J = this.numJoints;
    const i0 = inPose * J;
    const o0 = outPose * J;
    setTf(out, o0, tmul(rootTx, getTf(local, i0)));
    for (let j = 1; j < J; j++) {
      setTf(out, o0 + j, tmul(getTf(out, o0 + this.parents[j]), getTf(local, i0 + j)));
    }
    return out;
  }

  static fromJSON(data) {
    return new Skeleton(data.joint_names, data.parents, data.reference_local.flat());
  }
}
