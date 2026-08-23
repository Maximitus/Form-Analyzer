import type { ClubHeadEstimate, Point2D } from './types';

export interface ClubSearchLandmarks {
  leadWrist: Point2D | null;
  trailWrist: Point2D | null;
  midHip: Point2D | null;
  midShoulder: Point2D | null;
  head: Point2D | null;
  midAnkle: Point2D | null;
}

function hypot(x: number, y: number): number {
  return Math.hypot(x, y);
}

function normalize(x: number, y: number): Point2D | null {
  const mag = hypot(x, y);
  if (mag < 1e-6) return null;
  return {x: x / mag, y: y / mag};
}

export function gripPoint(landmarks: ClubSearchLandmarks): Point2D | null {
  return landmarks.trailWrist ?? landmarks.leadWrist;
}

export function shaftDirection(landmarks: ClubSearchLandmarks): Point2D | null {
  const grip = gripPoint(landmarks);
  if (!grip) return null;

  let gx = 0;
  let gy = 0;
  if (landmarks.leadWrist && landmarks.trailWrist) {
    gx = landmarks.trailWrist.x - landmarks.leadWrist.x;
    gy = landmarks.trailWrist.y - landmarks.leadWrist.y;
  }
  const gripLen = hypot(gx, gy);

  let tx = 0;
  let ty = 0;
  if (landmarks.midHip) {
    tx = grip.x - landmarks.midHip.x;
    ty = grip.y - landmarks.midHip.y;
  }
  const torsoLen = hypot(tx, ty);

  const gripWeight = gripLen > 12 ? 2.2 : gripLen > 4 ? 0.8 : 0.15;
  const torsoWeight = torsoLen > 8 ? 0.7 : 0.2;
  const downWeight = 1.0;

  return normalize(gx * gripWeight + tx * torsoWeight, gy * gripWeight + ty * torsoWeight + downWeight);
}

export function bodyHeightPx(landmarks: ClubSearchLandmarks): number | null {
  const top = landmarks.head ?? landmarks.midShoulder;
  const bottom = landmarks.midAnkle ?? landmarks.midHip;
  if (!top || !bottom) return null;
  const h = hypot(bottom.x - top.x, bottom.y - top.y);
  if (h < 8) return null;
  if (!landmarks.head && landmarks.midShoulder) return h * 1.25;
  return h;
}

export function kinematicClubHead(
  grip: Point2D,
  direction: Point2D,
  bodyHeight: number,
  lengthRatio = 0.58,
): Point2D {
  const length = bodyHeight * lengthRatio;
  return {
    x: grip.x + direction.x * length,
    y: grip.y + direction.y * length,
  };
}

function luma(data: Uint8ClampedArray | Uint8Array, width: number, x: number, y: number): number {
  const i = (y * width + x) * 4;
  return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
}

function windowStats(
  data: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  cx: number,
  cy: number,
  radius: number,
): {mean: number; std: number; grad: number} | null {
  const x0 = Math.round(cx);
  const y0 = Math.round(cy);
  let sum = 0;
  let sum2 = 0;
  let grad = 0;
  let n = 0;
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const x = x0 + dx;
      const y = y0 + dy;
      if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1) continue;
      const L = luma(data, width, x, y);
      sum += L;
      sum2 += L * L;
      n += 1;
      const gx = luma(data, width, x + 1, y) - luma(data, width, x - 1, y);
      const gy = luma(data, width, x, y + 1) - luma(data, width, x, y - 1);
      grad += Math.hypot(gx, gy);
    }
  }
  if (n < 4) return null;
  const mean = sum / n;
  const variance = Math.max(0, sum2 / n - mean * mean);
  return {mean, std: Math.sqrt(variance), grad: grad / n};
}

export function estimateClubHead(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  landmarks: ClubSearchLandmarks,
  prev: Point2D | null = null,
): ClubHeadEstimate | null {
  const grip = gripPoint(landmarks);
  const direction = shaftDirection(landmarks);
  const heightPx = bodyHeightPx(landmarks);
  if (!grip || !direction || heightPx === null) return null;

  const prior = kinematicClubHead(grip, direction, heightPx);
  const tMin = heightPx * 0.28;
  const tMax = heightPx * 0.95;
  const step = Math.max(3, Math.round(heightPx * 0.02));

  let best: {x: number; y: number; score: number} | null = null;

  for (let t = tMin; t <= tMax; t += step) {
    const x = grip.x + direction.x * t;
    const y = grip.y + direction.y * t;
    if (x < 2 || y < 2 || x >= width - 2 || y >= height - 2) continue;

    if (landmarks.midHip) {
      const bodyDist = hypot(x - landmarks.midHip.x, y - landmarks.midHip.y);
      if (bodyDist < heightPx * 0.18) continue;
    }

    const stats = windowStats(rgba, width, height, x, y, 4);
    if (!stats) continue;

    const contrast = Math.min(1, stats.std / 38);
    const dark = 1 - stats.mean / 255;
    const edge = Math.min(1, stats.grad / 70);
    const distal = (t - tMin) / Math.max(1e-3, tMax - tMin);
    let score = 0.28 * contrast + 0.18 * dark + 0.34 * edge + 0.2 * distal;
    if (prev) {
      const d = hypot(x - prev.x, y - prev.y);
      score += 0.18 * Math.exp(-d / (heightPx * 0.12));
    }

    if (!best || score > best.score) {
      best = {x, y, score};
    }
  }

  if (best && best.score >= 0.24) {
    return {
      x: best.x,
      y: best.y,
      score: Math.min(1, best.score),
      gripX: grip.x,
      gripY: grip.y,
      method: 'image',
    };
  }

  return {
    x: prior.x,
    y: prior.y,
    score: 0.16,
    gripX: grip.x,
    gripY: grip.y,
    method: 'prior',
  };
}
