import type {ClubSearchLandmarks} from './clubHead';
import {COCO_INDEX} from './metrics';
import type {GolfHandedness, Keypoint2D} from './types';
import {leadSideFor} from './metrics';

export const COCO_KEYPOINT_COUNT = 17;
export const COCO_KEYPOINT_NAMES = [
  'nose',
  'left_eye',
  'right_eye',
  'left_ear',
  'right_ear',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
] as const;

/**
 * Convert Form Analyzer tracked keypoints (COCO-style ids in app order) to a
 * dense COCO-17 array in pixel (or overlay) coordinates.
 */
export function trackedPoseToCoco(
  keypoints: readonly {x: number; y: number; v: number}[],
  trackedIds: readonly number[],
  scaleX = 1,
  scaleY = 1,
): Keypoint2D[] {
  const out: Keypoint2D[] = Array.from({length: COCO_KEYPOINT_COUNT}, (_, i) => ({
    x: 0,
    y: 0,
    score: 0,
    name: COCO_KEYPOINT_NAMES[i] ?? `kp${i}`,
  }));

  trackedIds.forEach((cocoId, i) => {
    if (cocoId < 0 || cocoId >= COCO_KEYPOINT_COUNT) return;
    const p = keypoints[i];
    if (!p) return;
    out[cocoId] = {
      x: p.x * scaleX,
      y: p.y * scaleY,
      score: p.v,
      name: COCO_KEYPOINT_NAMES[cocoId],
    };
  });
  return out;
}

export function clubLandmarksFromCoco(
  keypoints: readonly Keypoint2D[],
  handedness: GolfHandedness,
  minScore = 0.3,
): ClubSearchLandmarks {
  const take = (index: number) => {
    const p = keypoints[index];
    if (!p || p.score < minScore) return null;
    return {x: p.x, y: p.y};
  };
  const mid = (a: {x: number; y: number} | null, b: {x: number; y: number} | null) =>
    a && b ? {x: (a.x + b.x) / 2, y: (a.y + b.y) / 2} : a ?? b;

  const leadIsLeft = leadSideFor(handedness) === 'left';
  const leftWrist = take(COCO_INDEX.leftWrist);
  const rightWrist = take(COCO_INDEX.rightWrist);
  return {
    leadWrist: leadIsLeft ? leftWrist : rightWrist,
    trailWrist: leadIsLeft ? rightWrist : leftWrist,
    midHip: mid(take(COCO_INDEX.leftHip), take(COCO_INDEX.rightHip)),
    midShoulder: mid(take(COCO_INDEX.leftShoulder), take(COCO_INDEX.rightShoulder)),
    head: take(COCO_INDEX.nose),
    midAnkle: mid(take(COCO_INDEX.leftAnkle), take(COCO_INDEX.rightAnkle)),
  };
}

export function mapPoint(
  x: number,
  y: number,
  mapper: (xn: number, yn: number) => {x: number; y: number},
  srcW: number,
  srcH: number,
): {x: number; y: number} {
  if (srcW <= 0 || srcH <= 0) return {x, y};
  return mapper(x / srcW, y / srcH);
}

export function mapMetricsToOverlay(
  metrics: import('./types').GolfFrontalMetrics,
  mapper: (xn: number, yn: number) => {x: number; y: number},
  srcW: number,
  srcH: number,
): import('./types').GolfFrontalMetrics {
  const mapKp = (p: Keypoint2D | null) => {
    if (!p) return null;
    return {...p, ...mapPoint(p.x, p.y, mapper, srcW, srcH)};
  };
  const club = metrics.clubHead
    ? {
        ...metrics.clubHead,
        ...mapPoint(metrics.clubHead.x, metrics.clubHead.y, mapper, srcW, srcH),
        ...(() => {
          const g = mapPoint(metrics.clubHead.gripX, metrics.clubHead.gripY, mapper, srcW, srcH);
          return {gripX: g.x, gripY: g.y};
        })(),
      }
    : null;
  return {
    ...metrics,
    clubHead: club,
    landmarks: {
      midShoulder: mapKp(metrics.landmarks.midShoulder),
      midHip: mapKp(metrics.landmarks.midHip),
      midAnkle: mapKp(metrics.landmarks.midAnkle),
      midWrist: mapKp(metrics.landmarks.midWrist),
      head: mapKp(metrics.landmarks.head),
    },
  };
}
