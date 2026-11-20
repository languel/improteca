"""Golden fixtures from NVIDIA's own MotionBricks SOMA pipeline, run on CPU.

This runs the exact Warp kernels of NVIDIA's runtime (encode, decode, endpoint
correction, history) on the CPU, with ONNX Runtime in place of TensorRT. It
mirrors MotionBricksEngine.generate_animation() and the Inbetweener demo's
step(), with model_to_runtime set to the identity: our world is Y up, like
SOMA model space and three.js.

The fixtures are derived from NVIDIA's evaluation-only sample clips, so they
are written to local/ and never committed.

Usage (from the improteca root):
    python -I reference/make_fixtures.py <motionbricks checkout> <onnx> <clips dir> <out.json>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    repo, onnx_path, clips_dir, out_path = (Path(a) for a in sys.argv[1:5])
    sys.path.insert(0, str(repo))

    import onnxruntime as ort
    import warp as wp

    wp.config.quiet = True
    wp.init()
    wp.set_device("cpu")

    from motionbricks.io.bvh import load_bvh
    from motionbricks.io.usd import load_skeleton_from_usd
    from motionbricks.io.utils import SOMA_SKELETON_PRIM_PATH
    from motionbricks.model.config import MotionBricksInferenceConfig
    from motionbricks.model.contract import (
        MODEL_CONTEXT_AND_TARGET_POSES,
        MODEL_MAX_VALID_FRAMES,
        MODEL_POSES_PER_TOKEN,
        MODEL_SAMPLE_RATE,
    )
    from motionbricks.model.kernels import write_motionbricks_conditions_kernel
    from motionbricks.model.soma import kernels
    from motionbricks.model.soma.buffers import PoseToken
    from motionbricks.utils.pose_utils import compute_world_poses

    manifest = json.loads((repo / "motionbricks/assets/onnx/soma/soma_n1.json").read_text())
    joint_names = manifest["joint_names"]

    skeleton = load_skeleton_from_usd(
        str(repo / "motionbricks/assets/human/soma_base_skel_minimal_no_root.usd"), SOMA_SKELETON_PRIM_PATH,
    )
    J = skeleton.num_joints
    parents = skeleton.parent_indices.numpy().astype(int).tolist()
    names = list(skeleton.joint_names)

    model_to_pose = [names.index(n) for n in joint_names]
    assert model_to_pose[0] == 0
    pose_to_model = np.full(J, -1, dtype=np.int32)
    for model_joint, pose_joint in enumerate(model_to_pose):
        parent = parents[pose_joint]
        if parent == -1 or pose_to_model[parent] >= 0:
            pose_to_model[pose_joint] = model_joint
    right_leg, left_leg = names.index("RightLeg"), names.index("LeftLeg")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def tf_list(arr):
        return np.asarray(arr.numpy(), dtype=np.float64).reshape(-1, 7).round(7).tolist()

    def placement(x, z, yaw_deg):
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(yaw_deg))
        return wp.transform(wp.vec3(x, 0.0, z), q)

    def sample_window(clip, sample_time, place, anchor):
        """sample_pose_window() from the demo, for a Y-up world (ground is XZ)."""
        poses = wp.empty((MODEL_POSES_PER_TOKEN, J), dtype=wp.transform)
        clip.sample(sample_time, poses[anchor])
        root = poses[anchor].numpy().reshape(-1)
        ground = wp.transform(wp.vec3(-float(root[0]), 0.0, -float(root[2])), wp.quat_identity())
        root_offset = wp.mul(place, ground)
        for frame in range(MODEL_POSES_PER_TOKEN):
            t = sample_time + (frame - anchor) / MODEL_SAMPLE_RATE
            clip.sample(t, poses[frame], root_offset=root_offset)
        return poses

    def generate(context: PoseToken, target: PoseToken, config: MotionBricksInferenceConfig):
        guide_world = wp.empty((MODEL_CONTEXT_AND_TARGET_POSES, J), dtype=wp.transform)
        compute_world_poses(context.poses, skeleton.parent_indices, guide_world[:4], wp.transform_identity())
        compute_world_poses(target.poses, skeleton.parent_indices, guide_world[4:], wp.transform_identity())
        alignment = wp.empty(1, dtype=wp.transform)
        wp.launch(kernels.compute_guide_alignment_kernel, dim=1,
                  inputs=[right_leg, left_leg, wp.vec2(*config.heading_alignment), guide_world], outputs=[alignment])
        g = wp.zeros((1, 8, 5), dtype=wp.float32)
        l = wp.zeros((1, 8, 4), dtype=wp.float32)
        p = wp.zeros((1, 8, 267), dtype=wp.float32)
        wp.launch(kernels.encode_motionbricks_inputs_kernel, dim=MODEL_CONTEXT_AND_TARGET_POSES,
                  inputs=[len(joint_names), right_leg, left_leg, wp.array(model_to_pose, dtype=wp.int32),
                          guide_world, alignment], outputs=[g, l, p])
        hg = wp.zeros((1, 8), dtype=wp.int32)
        hl = wp.zeros((1, 8), dtype=wp.int32)
        hp = wp.zeros((1, 8), dtype=wp.int32)
        allowed = wp.zeros((1, 11), dtype=wp.int32)
        wp.launch(write_motionbricks_conditions_kernel, dim=11,
                  inputs=[target.num_poses, config.allowed_token_bits()], outputs=[hg, hl, hp, allowed])
        inputs = {
            "global_root_values": g.numpy(), "local_root_values": l.numpy(), "local_poses": p.numpy(),
            "has_global_root_values": hg.numpy(), "has_local_root_values": hl.numpy(),
            "has_local_poses": hp.numpy(), "allowed_pred_num_tokens": allowed.numpy(),
        }
        out = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
        pred = wp.array(out["pred_global_motions"].astype(np.float32), dtype=wp.float32)
        num_tokens = wp.array(out["pred_num_tokens"].astype(np.int32), dtype=wp.int32)
        out_poses = wp.empty((MODEL_MAX_VALID_FRAMES, J), dtype=wp.transform)
        contacts = wp.zeros((MODEL_MAX_VALID_FRAMES, 4), dtype=wp.float32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.launch(kernels.decode_motionbricks_output_kernel, dim=MODEL_MAX_VALID_FRAMES,
                  inputs=[len(joint_names), J, skeleton.parent_indices, wp.array(pose_to_model, dtype=wp.int32),
                          skeleton.reference_local_transforms, alignment, wp.transform_identity(), pred, num_tokens],
                  outputs=[out_poses, contacts, count])
        n = int(count.numpy()[0])
        decoded = out_poses.numpy()[:n].copy()
        start_d = wp.empty(J, dtype=wp.transform)
        end_d = wp.empty(J, dtype=wp.transform)
        wp.launch(kernels.compute_endpoint_deltas_kernel, dim=J,
                  inputs=[count, context.poses, target.poses, out_poses], outputs=[start_d, end_d])
        wp.launch(kernels.apply_endpoint_deltas_kernel, dim=(MODEL_MAX_VALID_FRAMES, J),
                  inputs=[count, start_d, end_d], outputs=[out_poses])
        return {
            "alignment": np.asarray(alignment.numpy()).reshape(-1).tolist(),
            "inputs": {k: np.asarray(v).reshape(-1).round(7).tolist() for k, v in inputs.items()},
            "pred_num_tokens": int(out["pred_num_tokens"][0]),
            "pred_global_motions": out["pred_global_motions"].reshape(-1).round(7).tolist(),
            "frame_count": n,
            "decoded": np.asarray(decoded, dtype=np.float64).reshape(-1, 7).round(7).tolist(),
            "corrected": tf_list(out_poses[:n]),
            "contacts": contacts.numpy()[:n].round(7).tolist(),
        }, out_poses

    # A chain like the Inbetweener demo: Start, then targets in order, each
    # generation starting from the last four played frames of the previous one.
    clips = {
        "idle": "idle_loop_001__A268.bvh",
        "walk": "Neutral_walk_forward_002__A057.bvh",
        "jump": "jump_over_obstacle_0_5m_R_001__A415.bvh",
        "look": "neutral_looking_around_R_001__A542.bvh",
    }
    loaded = {k: load_bvh(str(clips_dir / v), skeleton) for k, v in clips.items()}
    plan = [
        {"clip": "idle", "time": 1.0, "x": 0.0, "z": 0.0, "yaw": 0.0, "tokens": [8, 14], "num_poses": 4, "early": 0},
        {"clip": "look", "time": 3.0, "x": 1.5, "z": 0.5, "yaw": 90.0, "tokens": [8, 14], "num_poses": 4, "early": 0},
        {"clip": "walk", "time": 2.0, "x": 3.0, "z": -1.0, "yaw": -45.0, "tokens": [6, 16], "num_poses": 2, "early": 5},
        {"clip": "jump", "time": 1.2, "x": 1.0, "z": -2.5, "yaw": 180.0, "tokens": [8, 14], "num_poses": 1, "early": 0},
    ]

    windows = []
    for i, entry in enumerate(plan):
        place = placement(entry["x"], entry["z"], entry["yaw"])
        win = sample_window(loaded[entry["clip"]], entry["time"], place, anchor=3 if i == 0 else 0)
        windows.append(win)

    context = PoseToken(J)
    target = PoseToken(J)
    context.update_from_poses(windows[0])
    generations = []
    for i, entry in enumerate(plan[1:], start=1):
        target.update_from_poses(windows[i][: entry["num_poses"]])
        config = MotionBricksInferenceConfig(allowed_token_counts=tuple(range(entry["tokens"][0], entry["tokens"][1] + 1)))
        context_used = tf_list(context.poses)
        target_used = tf_list(target.poses)
        result, out_poses = generate(context, target, config)
        result["context"] = context_used
        result["target"] = target_used
        end = max(MODEL_POSES_PER_TOKEN - 1, result["frame_count"] - 1 - entry["early"])
        played = end + 1
        context.update_from_poses(out_poses[played - MODEL_POSES_PER_TOKEN:played])
        result["played_frames"] = played
        generations.append(result)
        print(f"generation {i}: tokens={result['pred_num_tokens']} frames={result['frame_count']} played={played}")

    # Clip sampling checks for the BVH loader port.
    samples = {
        name: {
            "times": [0.0, 0.5, 1.2345, 2.0],
            "poses": [tf_list(_sample(clip, t, J, wp)) for t in [0.0, 0.5, 1.2345, 2.0]],
            "num_frames": clip.num_frames,
            "sample_rate": clip.sample_rate,
        }
        for name, clip in loaded.items()
    }

    fixture = {
        "source": "NVIDIA MotionBricks runtime kernels on CPU with ONNX Runtime",
        "onnx": onnx_path.name,
        "skeleton": {
            "joint_names": names,
            "parents": parents,
            "reference_local": tf_list(skeleton.reference_local_transforms),
        },
        "model_joint_names": joint_names,
        "clips": clips,
        "plan": plan,
        "windows": [tf_list(w) for w in windows],
        "samples": samples,
        "generations": generations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def _sample(clip, t, J, wp):
    pose = wp.empty(J, dtype=wp.transform)
    clip.sample(t, pose)
    return pose


if __name__ == "__main__":
    main()
