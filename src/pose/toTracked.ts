import { COCO_KEYPOINT_COUNT } from '../types/pose';
import type { PoseFrame } from '../types/pose';

export type TrackedKeypoint = { x: number; y: number; v: number };

/**
 * Convert an RTMPose (COCO-17) frame to Form Analyzer's normalized tracked pose.
 *
 * `trackedIds` are COCO-style overlay ids. Heel/toe extensions 17–20 are not
 * RTMPose keypoints; callers should use ankles for ground contact and must not
 * treat ids 17–20 as BlazePose feet.
 */
export function poseFrameToTracked(
  pose: PoseFrame,
  trackedIds: number[],
  minScore = 0.15,
): TrackedKeypoint[] {
  const { width, height } = pose;
  if (width <= 0 || height <= 0) {
    return trackedIds.map(() => ({ x: 0, y: 0, v: 0 }));
  }
  return trackedIds.map((id) => {
    if (id < 0 || id >= COCO_KEYPOINT_COUNT) return { x: 0, y: 0, v: 0 };
    const kp = pose.keypoints[id];
    if (!kp || kp.score < minScore) return { x: 0, y: 0, v: 0 };
    return { x: kp.x / width, y: kp.y / height, v: kp.score };
  });
}

export function trackedHasVisible(keypoints: readonly TrackedKeypoint[], minScore = 0.15): boolean {
  return keypoints.some((p) => p.v >= minScore);
}
