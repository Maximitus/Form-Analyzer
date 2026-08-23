import type { Keypoint2D, Point2D } from './types';

const ANGLE_MIN = 0;
const ANGLE_MAX = 180;

function isFinitePoint(p: Point2D | undefined | null): p is Point2D {
  return !!p && Number.isFinite(p.x) && Number.isFinite(p.y);
}

export function jointAngleDeg(
  a: Point2D | undefined | null,
  vertex: Point2D | undefined | null,
  c: Point2D | undefined | null,
): number | null {
  if (!isFinitePoint(a) || !isFinitePoint(vertex) || !isFinitePoint(c)) {
    return null;
  }

  const v1x = a.x - vertex.x;
  const v1y = a.y - vertex.y;
  const v2x = c.x - vertex.x;
  const v2y = c.y - vertex.y;

  const mag1 = Math.hypot(v1x, v1y);
  const mag2 = Math.hypot(v2x, v2y);
  if (mag1 < 1e-6 || mag2 < 1e-6) {
    return null;
  }

  const cos = (v1x * v2x + v1y * v2y) / (mag1 * mag2);
  const clamped = Math.min(1, Math.max(-1, cos));
  const deg = (Math.acos(clamped) * 180) / Math.PI;
  return Math.min(ANGLE_MAX, Math.max(ANGLE_MIN, deg));
}

export function midpoint(a: Point2D, b: Point2D): Point2D {
  return {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2};
}

export function scoredKeypoint(
  keypoints: readonly Keypoint2D[],
  index: number,
  minScore: number,
): Keypoint2D | null {
  const kp = keypoints[index];
  if (!kp || kp.score < minScore) return null;
  return kp;
}

export function formatAngle(value: number | null): string {
  return value === null ? '—' : `${value.toFixed(0)}°`;
}

export function formatSignedAngle(value: number | null): string {
  if (value === null) return '—';
  const rounded = Math.abs(value) < 0.5 ? 0 : value;
  const sign = rounded > 0 ? '+' : '';
  return `${sign}${rounded.toFixed(0)}°`;
}

/** Positive = anatomical left side higher (smaller y). */
export function tiltFromHorizontalDeg(
  left: Point2D | undefined | null,
  right: Point2D | undefined | null,
): number | null {
  if (!isFinitePoint(left) || !isFinitePoint(right)) return null;
  const dx = Math.abs(right.x - left.x);
  const dy = right.y - left.y;
  if (Math.hypot(dx, dy) < 1e-6) return null;
  const deg = (Math.atan2(dy, dx) * 180) / Math.PI;
  return Math.min(90, Math.max(-90, deg));
}

/** Positive = superior landmark toward image +x. */
export function leanFromVerticalDeg(
  superior: Point2D | undefined | null,
  inferior: Point2D | undefined | null,
): number | null {
  if (!isFinitePoint(superior) || !isFinitePoint(inferior)) return null;
  const dx = superior.x - inferior.x;
  const up = inferior.y - superior.y;
  if (Math.hypot(dx, up) < 1e-6) return null;
  const deg = (Math.atan2(dx, up) * 180) / Math.PI;
  return Math.min(90, Math.max(-90, deg));
}
