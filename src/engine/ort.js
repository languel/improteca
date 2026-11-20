// Run the MotionBricks ONNX model with ONNX Runtime. `ort` is either
// onnxruntime-web (browser, WebGPU or WebAssembly) or onnxruntime-node
// (tests, CPU); both share this API.

const SHAPES = {
  global_root_values: ['float32', [1, 8, 5]],
  local_root_values: ['float32', [1, 8, 4]],
  local_poses: ['float32', [1, 8, 267]],
  has_global_root_values: ['int32', [1, 8]],
  has_local_root_values: ['int32', [1, 8]],
  has_local_poses: ['int32', [1, 8]],
  allowed_pred_num_tokens: ['int32', [1, 11]],
};

/**
 * @param {any} ort  the onnxruntime module
 * @param {any} session  an InferenceSession for mmm_v6_soma_*.onnx
 */
export function ortRunner(ort, session) {
  return async (inputs) => {
    const feeds = {};
    for (const [name, [type, dims]] of Object.entries(SHAPES)) {
      feeds[name] = new ort.Tensor(type, inputs[name], dims);
    }
    const out = await session.run(feeds);
    return {
      pred_global_motions: out.pred_global_motions.data,
      pred_num_tokens: out.pred_num_tokens.data,
    };
  };
}

/** NVIDIA's ONNX files, served with CORS from GitHub's LFS media host. */
export const MODEL_URLS = {
  fp16: 'https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets/onnx/soma/mmm_v6_soma_fp16.onnx',
  fp32: 'https://media.githubusercontent.com/media/NVlabs/motionbricks/main/motionbricks/assets/onnx/soma/mmm_v6_soma_fp32.onnx',
};

/** SHA-256 of the FP16 file used to make the reference fixtures. */
export const FP16_SHA256 = '6c9f78e23decbda8e1cfd1b0cc96980ea4b24b3ca2c808d03b058f48f82fc9ef';
