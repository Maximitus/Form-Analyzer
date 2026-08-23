/**
 * SimCC (Simple Coordinate Classification) decode for RTMPose.
 *
 * Each keypoint is represented by two 1-D logit vectors (X and Y). Peak
 * location via argmax gives the integer bin; a local softmax around that
 * peak yields a sub-pixel expectation. Coordinates are in model-input
 * pixels after dividing by `splitRatio` (typically 2.0).
 */

export interface SimccDecodeOptions {
  splitRatio?: number;
  /** Odd window used for local softmax sub-pixel refinement. */
  softWindow?: number;
}

export interface DecodedSimcc {
  xs: Float32Array;
  ys: Float32Array;
  scores: Float32Array;
}

function argmax1d(data: Float32Array, offset: number, length: number): { index: number; value: number } {
  let maxIndex = 0;
  let maxValue = -Infinity;
  for (let i = 0; i < length; i++) {
    const v = data[offset + i];
    if (v > maxValue) {
      maxValue = v;
      maxIndex = i;
    }
  }
  return { index: maxIndex, value: maxValue };
}

/**
 * Softmax-weighted expected bin index over a window centered on `peakIndex`.
 */
export function softArgmax1d(
  data: Float32Array,
  offset: number,
  length: number,
  peakIndex: number,
  window: number,
): { coord: number; mass: number } {
  const half = Math.max(0, Math.floor(window / 2));
  const start = Math.max(0, peakIndex - half);
  const end = Math.min(length - 1, peakIndex + half);

  let maxLogit = -Infinity;
  for (let i = start; i <= end; i++) {
    const v = data[offset + i];
    if (v > maxLogit) maxLogit = v;
  }

  let sum = 0;
  let weighted = 0;
  for (let i = start; i <= end; i++) {
    const e = Math.exp(data[offset + i] - maxLogit);
    sum += e;
    weighted += e * i;
  }

  if (sum <= 0 || !Number.isFinite(sum)) {
    return { coord: peakIndex, mass: 0 };
  }

  return { coord: weighted / sum, mass: sum };
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function planeLayout(dims: readonly number[]): { batch: number; keys: number; bins: number } {
  if (dims.length === 1) {
    return { batch: 1, keys: 1, bins: dims[0] };
  }
  if (dims.length === 2) {
    return { batch: 1, keys: dims[0], bins: dims[1] };
  }
  const bins = dims[dims.length - 1];
  const keys = dims[dims.length - 2];
  const batch = dims.slice(0, -2).reduce((a, b) => a * b, 1);
  return { batch, keys, bins };
}

/**
 * Decode SimCC X/Y logit tensors into sub-pixel keypoints and confidences.
 * `simccX` / `simccY` are row-major Float32 arrays with shape [N, K, bins].
 */
export function decodeSimcc(
  simccX: Float32Array,
  simccXDims: readonly number[],
  simccY: Float32Array,
  simccYDims: readonly number[],
  options: SimccDecodeOptions = {},
): DecodedSimcc {
  const splitRatio = options.splitRatio ?? 2;
  const softWindow = options.softWindow ?? 11;

  const xLayout = planeLayout(simccXDims);
  const yLayout = planeLayout(simccYDims);
  const keyCount = Math.min(xLayout.keys, yLayout.keys);
  const personCount = Math.min(xLayout.batch, yLayout.batch);
  const total = personCount * keyCount;

  const xs = new Float32Array(total);
  const ys = new Float32Array(total);
  const scores = new Float32Array(total);

  for (let n = 0; n < personCount; n++) {
    for (let k = 0; k < keyCount; k++) {
      const outIndex = n * keyCount + k;
      const xOffset = (n * xLayout.keys + k) * xLayout.bins;
      const yOffset = (n * yLayout.keys + k) * yLayout.bins;

      const peakX = argmax1d(simccX, xOffset, xLayout.bins);
      const peakY = argmax1d(simccY, yOffset, yLayout.bins);

      const softX = softArgmax1d(simccX, xOffset, xLayout.bins, peakX.index, softWindow);
      const softY = softArgmax1d(simccY, yOffset, yLayout.bins, peakY.index, softWindow);

      xs[outIndex] = softX.coord / splitRatio;
      ys[outIndex] = softY.coord / splitRatio;

      // MMPose get_simcc_maximum uses min(max_x, max_y); map to (0, 1) via sigmoid.
      const raw = Math.min(peakX.value, peakY.value);
      scores[outIndex] = Math.min(1, Math.max(0, sigmoid(raw)));
    }
  }

  return { xs, ys, scores };
}

/**
 * Identify which output is SimCC-X vs SimCC-Y from last-dimension bin counts.
 * For 192×256 input with split ratio 2: X bins = 384, Y bins = 512.
 */
export function identifySimccOutputs(
  outputs: Array<{ name: string; dims: readonly number[] }>,
  inputWidth: number,
  inputHeight: number,
  splitRatio: number,
): { xName: string; yName: string } {
  if (outputs.length < 2) {
    throw new Error('RTMPose SimCC models must emit at least two outputs');
  }

  const expectedX = inputWidth * splitRatio;
  const expectedY = inputHeight * splitRatio;

  const lastDim = (dims: readonly number[]): number =>
    dims.length === 0 ? 0 : (dims[dims.length - 1] ?? 0);

  const byLastDim = (dim: number) => outputs.find((o) => lastDim(o.dims) === dim);

  const namedX = outputs.find((o) => /simcc[_]?x/i.test(o.name));
  const namedY = outputs.find((o) => /simcc[_]?y/i.test(o.name));
  if (namedX && namedY) {
    return { xName: namedX.name, yName: namedY.name };
  }

  const matchX = byLastDim(expectedX);
  const matchY = byLastDim(expectedY);
  if (matchX && matchY && matchX.name !== matchY.name) {
    return { xName: matchX.name, yName: matchY.name };
  }

  const sorted = [...outputs].sort((a, b) => lastDim(a.dims) - lastDim(b.dims));
  return { xName: sorted[0].name, yName: sorted[sorted.length - 1].name };
}
