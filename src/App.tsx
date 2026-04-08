/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  useState,
  useRef,
  useEffect,
  useLayoutEffect,
  useCallback,
  ChangeEvent,
  MutableRefObject,
  PointerEvent as ReactPointerEvent,
  type ReactNode,
} from 'react';
import {
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Hand,
  Maximize,
  Minimize,
  Minus,
  Trash,
  ZoomIn,
  ZoomOut,
  Camera,
  Image as ImageGalleryIcon,
  Images,
  Video,
  Activity,
  LineChart,
} from 'lucide-react';
import * as poseDetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import SettingsMenu from './SettingsMenu';
import { useTheme } from './theme';

/** Toolbar icon: two rays meeting at a vertex (angle measure). */
function AngleMeasureIcon({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden
    >
      <path d="M6 18 12 6 18 18" />
    </svg>
  );
}

function pickVideoRecorderMimeType(): string | undefined {
  if (typeof MediaRecorder === 'undefined') return undefined;
  if (typeof MediaRecorder.isTypeSupported !== 'function') return undefined;
  const candidates = [
    'video/webm;codecs=vp9,opus',
    'video/webm;codecs=vp8,opus',
    'video/webm',
    'video/mp4',
  ];
  for (const t of candidates) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return undefined;
}

/**
 * MediaRecorder WebM/MP4 blobs often report a bogus `duration` (e.g. 10× too large) while
 * `seekable` already matches the real timeline. Prefer seekable when they disagree.
 */
function getReliableVideoDuration(video: HTMLVideoElement): number {
  let seekableEnd = 0;
  try {
    if (video.seekable && video.seekable.length > 0) {
      seekableEnd = video.seekable.end(video.seekable.length - 1);
    }
  } catch {
    seekableEnd = 0;
  }

  const raw = video.duration;
  const durOk = Number.isFinite(raw) && raw > 0 && raw !== Number.POSITIVE_INFINITY;

  if (seekableEnd > 0) {
    if (!durOk || (durOk && Math.abs(raw - seekableEnd) / seekableEnd > 0.35)) {
      return seekableEnd;
    }
  }

  if (durOk && raw < 86400 * 365) {
    return raw;
  }
  if (seekableEnd > 0) {
    return seekableEnd;
  }
  return 0;
}

const ZOOM_MIN_SCALE = 1;
const ZOOM_MAX_SCALE = 8;
const ZOOM_BUTTON_FACTOR = 1.25;
const FRAME_STEP_SECONDS = 1 / 30; // approx single frame at 30fps
const SLIDER_SCRUB_STEP_SECONDS = 1 / 60; // smaller increments for smoother scrubbing
/**
 * COCO-style IDs kept for all math and overlays. 17–20 are our extensions for
 * heel / toe (BlazePose only); MoveNet had no foot landmarks beyond the ankle.
 */
const POSE_NOSE_ID = 0;
const POSE_TRACKED_IDS = [
  POSE_NOSE_ID,
  11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10, 17, 18, 19, 20,
];
/** Maps each COCO-style id to BlazePose keypoint index (model output order). */
const COCO_ID_TO_BLAZEPOSE_INDEX: Record<number, number> = {
  0: 0,
  5: 11,
  6: 12,
  7: 13,
  8: 14,
  9: 15,
  10: 16,
  11: 23,
  12: 24,
  13: 25,
  14: 26,
  15: 27,
  16: 28,
  17: 29,
  18: 30,
  19: 31,
  20: 32,
};
const POSE_LEFT_SIDE_IDS = new Set([5, 7, 9, 11, 13, 15, 17, 19]);
const POSE_RIGHT_SIDE_IDS = new Set([6, 8, 10, 12, 14, 16, 18, 20]);
const POSE_ARM_IDS = new Set([7, 8, 9, 10]);
const POSE_BODY_CONNECTIONS: [number, number][] = [
  [POSE_NOSE_ID, 5],
  [POSE_NOSE_ID, 6],
  [11, 13],
  [13, 15],
  [15, 17],
  [17, 19],
  [12, 14],
  [14, 16],
  [16, 18],
  [18, 20],
  [11, 12],
  [5, 11],
  [6, 12],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
];
const MIN_POSE_VISIBILITY = 0.25;
const KNEE_SAMPLE_EPSILON_SECONDS = 1 / 1000;
const POSE_INFERENCE_HZ = 24;

function keypointsFromBlazePoseOutput(
  keypoints: poseDetection.Keypoint[] | undefined,
  sourceWidth: number,
  sourceHeight: number,
): {x: number; y: number; v: number}[] {
  if (!keypoints || keypoints.length === 0 || sourceWidth <= 0 || sourceHeight <= 0) {
    return [];
  }
  return POSE_TRACKED_IDS.map((cocoId) => {
    const bpIdx = COCO_ID_TO_BLAZEPOSE_INDEX[cocoId];
    const p = keypoints[bpIdx];
    if (!p) return {x: 0, y: 0, v: 0};
    const score = p.score ?? 0;
    return {x: p.x / sourceWidth, y: p.y / sourceHeight, v: score};
  });
}

/**
 * BlazePose TF.js uses getImageSize(input) which reads HTMLVideoElement.width/height,
 * not videoWidth/videoHeight — those are often 0 or wrong, collapsing landmarks.
 * Copy the current frame to a canvas sized to the intrinsic video dimensions so
 * the model and our normalization share one coordinate system.
 */
function blazeposeVideoFrameCanvas(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): HTMLCanvasElement | null {
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (vw <= 0 || vh <= 0) return null;
  if (canvas.width !== vw || canvas.height !== vh) {
    canvas.width = vw;
    canvas.height = vh;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, vw, vh);
  return canvas;
}
const KNEE_MAXIMA_MIN_SAMPLES = 3;
const KNEE_MAXIMA_MIN_GAP_SECONDS = 0.15;
/** Min prominence (°): peak must stand this far above the higher adjacent "valley" baseline (filters slope noise). */
const KNEE_MAXIMA_MIN_PROMINENCE = 5;
/** Peak must be >= every sample in ±radius (stricter than single-neighbor local max). */
const KNEE_MAXIMA_NEIGHBOR_RADIUS = 2;
const KNEE_MINIMA_MIN_SAMPLES = 3;
const KNEE_MINIMA_MIN_GAP_SECONDS = 0.3;
const KNEE_MINIMA_DEFAULT_PROMINENCE = 15;
/** Stride: first sample after peak where knee bends this much below peak angle ≈ landing flexion (earlier than mid-stance min). */
const STRIDE_FLEX_ONSET_DEG = 2;
/** Search only the first ~120ms after extension peak for contact (avoids late mid-stance). */
const STRIDE_CONTACT_SEARCH_SEC = 0.12;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function calculateJointAngle(a: {x: number; y: number}, b: {x: number; y: number}, c: {x: number; y: number}, aspectRatio = 1) {
  const angle1 = Math.atan2(a.y - b.y, (a.x - b.x) * aspectRatio);
  const angle2 = Math.atan2(c.y - b.y, (c.x - b.x) * aspectRatio);
  let angle = Math.abs((angle1 - angle2) * (180 / Math.PI));
  if (angle > 180) angle = 360 - angle;
  return angle;
}

function getKneeAngleFromPose(
  keypoints: {x: number; y: number; v: number}[],
  side: KneeSide,
  visibilityThreshold = MIN_POSE_VISIBILITY,
  aspectRatio = 1,
) {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });

  const leftHip = byId.get(11);
  const leftKnee = byId.get(13);
  const leftAnkle = byId.get(15);
  const rightHip = byId.get(12);
  const rightKnee = byId.get(14);
  const rightAnkle = byId.get(16);

  if (side === 'left') {
    const leftVisible =
      !!leftHip &&
      !!leftKnee &&
      !!leftAnkle &&
      Math.min(leftHip.v, leftKnee.v, leftAnkle.v) >= visibilityThreshold;
    if (!leftVisible) return null;
    return calculateJointAngle(leftHip!, leftKnee!, leftAnkle!, aspectRatio);
  }

  const rightVisible =
    !!rightHip &&
    !!rightKnee &&
    !!rightAnkle &&
    Math.min(rightHip.v, rightKnee.v, rightAnkle.v) >= visibilityThreshold;
  if (!rightVisible) return null;
  return calculateJointAngle(rightHip!, rightKnee!, rightAnkle!, aspectRatio);
}

function pickKneeTrackingSide(
  keypoints: {x: number; y: number; v: number}[],
  visibilityThreshold = MIN_POSE_VISIBILITY,
): KneeSide | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });

  const leftHip = byId.get(11);
  const leftKnee = byId.get(13);
  const leftAnkle = byId.get(15);
  const rightHip = byId.get(12);
  const rightKnee = byId.get(14);
  const rightAnkle = byId.get(16);

  const leftScore =
    leftHip && leftKnee && leftAnkle ? Math.min(leftHip.v, leftKnee.v, leftAnkle.v) : 0;
  const rightScore =
    rightHip && rightKnee && rightAnkle ? Math.min(rightHip.v, rightKnee.v, rightAnkle.v) : 0;
  if (leftScore < visibilityThreshold && rightScore < visibilityThreshold) return null;
  return leftScore >= rightScore ? 'left' : 'right';
}

function getTrackedLegAnchors(
  keypoints: {x: number; y: number; v: number}[],
  side: KneeSide,
  visibilityThreshold = MIN_POSE_VISIBILITY,
) {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const hip = side === 'left' ? byId.get(11) : byId.get(12);
  const ankle = side === 'left' ? byId.get(15) : byId.get(16);
  if (!hip || !ankle) return null;
  if (Math.min(hip.v, ankle.v) < visibilityThreshold) return null;
  return {hipX: hip.x, ankleX: ankle.x};
}

/** Angle from horizontal at the knee to the hip. 0° = parallel, negative = below parallel. */
function getSquatDepthAngle(
  keypoints: {x: number; y: number; v: number}[],
  side: KneeSide,
  visibilityThreshold = MIN_POSE_VISIBILITY,
  aspectRatio = 1,
): number | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const hip = side === 'left' ? byId.get(11) : byId.get(12);
  const knee = side === 'left' ? byId.get(13) : byId.get(14);
  if (!hip || !knee) return null;
  if (Math.min(hip.v, knee.v) < visibilityThreshold) return null;
  const dx = Math.abs(hip.x - knee.x) * aspectRatio;
  const dy = knee.y - hip.y;
  if (dx < 1e-9 && Math.abs(dy) < 1e-9) return null;
  return Math.atan2(dy, dx) * (180 / Math.PI);
}

function getBackAngle(
  keypoints: {x: number; y: number; v: number}[],
  side: KneeSide,
  visibilityThreshold = MIN_POSE_VISIBILITY,
  aspectRatio = 1,
): number | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const shoulder = side === 'left' ? byId.get(5) : byId.get(6);
  const hip = side === 'left' ? byId.get(11) : byId.get(12);
  if (!shoulder || !hip) return null;
  if (Math.min(shoulder.v, hip.v) < visibilityThreshold) return null;
  const dx = Math.abs(shoulder.x - hip.x) * aspectRatio;
  const dy = hip.y - shoulder.y;
  if (dx < 1e-9 && Math.abs(dy) < 1e-9) return null;
  // Report trunk inclination from vertical (biomechanics standard).
  return 90 - (Math.atan2(dy, dx) * (180 / Math.PI));
}

/** Angle at the hip joint: shoulder-hip-knee. */
function getHipAngle(
  keypoints: {x: number; y: number; v: number}[],
  side: KneeSide,
  visibilityThreshold = MIN_POSE_VISIBILITY,
  aspectRatio = 1,
): number | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const shoulder = side === 'left' ? byId.get(5) : byId.get(6);
  const hip = side === 'left' ? byId.get(11) : byId.get(12);
  const knee = side === 'left' ? byId.get(13) : byId.get(14);
  if (!shoulder || !hip || !knee) return null;
  if (Math.min(shoulder.v, hip.v, knee.v) < visibilityThreshold) return null;
  return calculateJointAngle(shoulder, hip, knee, aspectRatio);
}


/** Euclidean distance between two normalized keypoints, corrected for pixel aspect ratio. */
function segmentLength(
  a: {x: number; y: number},
  b: {x: number; y: number},
  aspectRatio = 1,
): number {
  const dx = (a.x - b.x) * aspectRatio;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

type BodyProportions = {
  torsoLen: number;
  femurLen: number;
  tibiaLen: number;
  femurToTorso: number;
  tibiaToFemur: number;
  legToTorso: number;
};

type ProportionProfile = {
  proportions: BodyProportions;
  femurCategory: 'short' | 'slightly-short' | 'average' | 'slightly-long' | 'long';
  tibiaCategory: 'short' | 'slightly-short' | 'average' | 'slightly-long' | 'long';
  /** 0 = easiest possible, 100 = hardest possible */
  difficultyScore: number;
  /** Estimated minimum torso forward lean (degrees from vertical) at parallel depth */
  estimatedLeanDeg: number;
  /** Estimated knee flexion at parallel depth (0 = fully extended) */
  estimatedParallelKneeFlexDeg: number;
  /** Estimated hip flexion proxy at parallel depth (trunk-thigh proxy) */
  estimatedParallelHipFlexDeg: number;
  dominantPattern: 'quad-dominant' | 'posterior-chain' | 'balanced';
  insights: string[];
};

function getBodyProportions(
  keypoints: {x: number; y: number; v: number}[],
  side: KneeSide,
  visibilityThreshold = MIN_POSE_VISIBILITY,
  aspectRatio = 1,
): BodyProportions | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const leftShoulder = byId.get(5);
  const rightShoulder = byId.get(6);
  const leftHip = byId.get(11);
  const rightHip = byId.get(12);
  const hip = side === 'left' ? leftHip : rightHip;
  const knee = side === 'left' ? byId.get(13) : byId.get(14);
  const ankle = side === 'left' ? byId.get(15) : byId.get(16);
  if (!leftShoulder || !rightShoulder || !leftHip || !rightHip || !hip || !knee || !ankle) return null;
  if (
    Math.min(
      leftShoulder.v,
      rightShoulder.v,
      leftHip.v,
      rightHip.v,
      hip.v,
      knee.v,
      ankle.v,
    ) < visibilityThreshold
  ) {
    return null;
  }

  // Mid-shoulder → mid-hip: stable trunk line vs one glenohumeral point. Ipsilateral
  // shoulder–hip stretches when the shoulder rolls forward in a squat, inflating
  // "torso" and biasing femur/torso toward "short femur."
  const midShoulder = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2,
  };
  const midHip = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2,
  };
  const torsoLen = segmentLength(midShoulder, midHip, aspectRatio);
  const femurLen = segmentLength(hip, knee, aspectRatio);
  const tibiaLen = segmentLength(knee, ankle, aspectRatio);

  if (torsoLen < 1e-6 || femurLen < 1e-6 || tibiaLen < 1e-6) return null;

  // Reject physiologically impossible proportions (detection errors, extreme
  // camera angles, or partial occlusion producing garbage geometry).
  // Each segment should be 18–50% of the total and no segment > 2× another.
  const total = torsoLen + femurLen + tibiaLen;
  const pcts = [torsoLen / total, femurLen / total, tibiaLen / total];
  if (pcts.some((p) => p < 0.18 || p > 0.50)) return null;
  const segs = [torsoLen, femurLen, tibiaLen];
  for (let i = 0; i < segs.length; i++) {
    for (let j = i + 1; j < segs.length; j++) {
      const ratio = segs[i]! / segs[j]!;
      if (ratio > 2.0 || ratio < 0.5) return null;
    }
  }

  return {
    torsoLen,
    femurLen,
    tibiaLen,
    femurToTorso: femurLen / torsoLen,
    tibiaToFemur: tibiaLen / femurLen,
    legToTorso: (femurLen + tibiaLen) / torsoLen,
  };
}

/**
 * Linear correction of raw 2D segment lengths from BlazePose so ratios sit closer to
 * joint-center anthropometry (Drillis-style). Low hip landmarks shorten measured femur
 * and inflate tibia/femur; midline torso is still a bit long vs true trunk height.
 * Category bands and the parallel lean model then use **calibrated** lengths.
 */
const POSE_SEGMENT_FEMUR_LEN_SCALE = 1.06;
const POSE_SEGMENT_TORSO_LEN_SCALE = 0.99;
const POSE_SEGMENT_TIBIA_LEN_SCALE = 1;

/**
 * Tape-measured reference (same units, any unit — only ratios matter). Tunes 2D pose
 * segment ratios toward real segment lengths without extra UI inputs.
 * Ref values: app “Proportions (average)” before this layer (~0.87 F/T, ~1.03 T/F).
 */
const VALIDATION_TORSO_LEN = 21;
const VALIDATION_FEMUR_LEN = 20;
const VALIDATION_TIBIA_LEN = 20;
const VALIDATION_REF_FEMUR_TO_TORSO = 0.87;
const VALIDATION_REF_TIBIA_TO_FEMUR = 1.03;

const VALIDATION_TARGET_FEMUR_TO_TORSO = VALIDATION_FEMUR_LEN / VALIDATION_TORSO_LEN;
const VALIDATION_TARGET_TIBIA_TO_FEMUR = VALIDATION_TIBIA_LEN / VALIDATION_FEMUR_LEN;
/** Multiply femur length vs torso so (femur/torso) moves ref → target. */
const RATIO_FIX_FEMUR_VS_TORSO = VALIDATION_TARGET_FEMUR_TO_TORSO / VALIDATION_REF_FEMUR_TO_TORSO;
/** Multiply tibia vs femur so (tibia/femur) moves ref → target (applied after femur fix). */
const RATIO_FIX_TIBIA_VS_FEMUR = VALIDATION_TARGET_TIBIA_TO_FEMUR / VALIDATION_REF_TIBIA_TO_FEMUR;

function calibrateBodyProportions(raw: BodyProportions): BodyProportions {
  const torsoLen = raw.torsoLen * POSE_SEGMENT_TORSO_LEN_SCALE;
  const femurLen =
    raw.femurLen * POSE_SEGMENT_FEMUR_LEN_SCALE * RATIO_FIX_FEMUR_VS_TORSO;
  const tibiaLen =
    raw.tibiaLen
    * POSE_SEGMENT_TIBIA_LEN_SCALE
    * RATIO_FIX_FEMUR_VS_TORSO
    * RATIO_FIX_TIBIA_VS_FEMUR;
  if (torsoLen < 1e-9 || femurLen < 1e-9 || tibiaLen < 1e-9) {
    return raw;
  }
  return {
    torsoLen,
    femurLen,
    tibiaLen,
    femurToTorso: femurLen / torsoLen,
    tibiaToFemur: tibiaLen / femurLen,
    legToTorso: (femurLen + tibiaLen) / torsoLen,
  };
}

/**
 * Estimate the minimum forward lean of the torso (degrees from vertical) needed
 * to keep center of mass over midfoot at parallel squat depth.
 *
 * 2D sagittal-plane balance model:
 *   - At parallel the femur is horizontal (hip drops to knee height).
 *   - The tibia tilts forward from the ankle by ankleDorsiflex (default 22°,
 *     calibrated against Fuglsang et al. 2017 JSCR data).
 *   - Knee-x  = ankle-x + tibia × sin(dorsiflex).
 *   - Hip-x   = knee-x − femur (femur horizontal).
 *   - The ankle is at the rear ~25% of the foot; balance is over the midfoot,
 *     which sits ~0.038H forward of the ankle (Drillis & Contini 1966).
 *     We estimate H from the three measured segments: H ≈ (T+F+S) / 0.779.
 *   - Balance: shoulder-x ≈ midfoot-x.
 *   → sin(lean) = (midfootOffset + femur − tibia × sin(dorsiflex)) / torso
 */
const MIDFOOT_RATIO = 0.038 / 0.779;
const DEFAULT_PARALLEL_ANKLE_DORSIFLEX_DEG = 22;
function estimateForwardLean(
  p: BodyProportions,
  ankleDorsiflexDeg = DEFAULT_PARALLEL_ANKLE_DORSIFLEX_DEG,
): number {
  const dorsRad = ankleDorsiflexDeg * (Math.PI / 180);
  const totalSegs = p.torsoLen + p.femurLen + p.tibiaLen;
  const midfootOffset = MIDFOOT_RATIO * totalSegs;
  const kneeForward = p.tibiaLen * Math.sin(dorsRad);
  const hipBehind = midfootOffset + p.femurLen - kneeForward;
  const sinLean = hipBehind / p.torsoLen;
  const clamped = Math.min(1, Math.max(-1, sinLean));
  return Math.asin(clamped) * (180 / Math.PI);
}

function estimateParallelFlexionAngles(
  p: BodyProportions,
  ankleDorsiflexDeg = DEFAULT_PARALLEL_ANKLE_DORSIFLEX_DEG,
): {kneeFlexDeg: number; hipFlexProxyDeg: number} {
  const leanDeg = estimateForwardLean(p, ankleDorsiflexDeg);
  // At parallel with femur horizontal in this model:
  // knee flexion = 90° + ankle dorsiflexion
  // hip flexion proxy = 90° + trunk lean-from-vertical
  const kneeFlexDeg = clamp(90 + ankleDorsiflexDeg, 0, 180);
  const hipFlexProxyDeg = clamp(90 + leanDeg, 0, 180);
  return {kneeFlexDeg, hipFlexProxyDeg};
}

/** Drillis-style population bands on **calibrated** femur/torso and tibia/femur ratios. */
const PROP_FEMUR_TORSO_SHORT = 0.78;
const PROP_FEMUR_TORSO_SLIGHTLY_SHORT = 0.82;
const PROP_FEMUR_TORSO_SLIGHTLY_LONG = 0.88;
const PROP_FEMUR_TORSO_LONG = 0.92;
const PROP_TIBIA_FEMUR_SHORT = 0.92;
const PROP_TIBIA_FEMUR_SLIGHTLY_SHORT = 0.96;
const PROP_TIBIA_FEMUR_SLIGHTLY_LONG = 1.04;
const PROP_TIBIA_FEMUR_LONG = 1.08;

function classifyProportions(raw: BodyProportions): ProportionProfile {
  const p = calibrateBodyProportions(raw);

  const femurCategory: ProportionProfile['femurCategory'] =
    p.femurToTorso < PROP_FEMUR_TORSO_SHORT ? 'short'
    : p.femurToTorso < PROP_FEMUR_TORSO_SLIGHTLY_SHORT ? 'slightly-short'
    : p.femurToTorso > PROP_FEMUR_TORSO_LONG ? 'long'
    : p.femurToTorso > PROP_FEMUR_TORSO_SLIGHTLY_LONG ? 'slightly-long'
    : 'average';
  const tibiaCategory: ProportionProfile['tibiaCategory'] =
    p.tibiaToFemur < PROP_TIBIA_FEMUR_SHORT ? 'short'
    : p.tibiaToFemur < PROP_TIBIA_FEMUR_SLIGHTLY_SHORT ? 'slightly-short'
    : p.tibiaToFemur > PROP_TIBIA_FEMUR_LONG ? 'long'
    : p.tibiaToFemur > PROP_TIBIA_FEMUR_SLIGHTLY_LONG ? 'slightly-long'
    : 'average';

  const estimatedLeanDeg = estimateForwardLean(p);
  const parallelFlex = estimateParallelFlexionAngles(p);

  // Continuous difficulty score (0–100).
  // Favorable proportions yield ~35°; challenging ones reach 55°+.
  // Map the lean range [30°, 55°] → [0, 100].
  const difficultyScore = clamp(((estimatedLeanDeg - 30) / 25) * 100, 0, 100);

  const dominantPattern: ProportionProfile['dominantPattern'] =
    estimatedLeanDeg > 48 ? 'posterior-chain'
    : estimatedLeanDeg < 38 ? 'quad-dominant'
    : 'balanced';

  const insights: string[] = [];

  // Headline insight based on the physics-derived lean from vertical.
  if (estimatedLeanDeg > 50) {
    insights.push(`Proportions require substantial forward lean (~${Math.round(estimatedLeanDeg)}° from vertical) to balance at parallel.`);
  } else if (estimatedLeanDeg > 42) {
    insights.push(`Moderate forward lean (~${Math.round(estimatedLeanDeg)}° from vertical) needed at parallel depth.`);
  } else {
    insights.push(`Favorable geometry — only ~${Math.round(estimatedLeanDeg)}° forward lean needed at parallel.`);
  }

  // Pattern-specific cues
  if (dominantPattern === 'posterior-chain') {
    insights.push('This lean angle loads the posterior chain heavily — expect to feel it in glutes and lower back.');
    insights.push('Squat shoes (heel elevation) reduce forward lean by ~5-8° and shift load toward quads.');
  } else if (dominantPattern === 'quad-dominant') {
    insights.push('A relatively upright torso means quads will be the primary mover.');
    insights.push('Lower back stress should be minimal; high-bar position is a natural fit.');
  }

  // Interaction-aware cues — treat "slightly-X" as leaning toward X with softer language.
  const femurLong = femurCategory === 'long' || femurCategory === 'slightly-long';
  const femurShort = femurCategory === 'short' || femurCategory === 'slightly-short';
  const tibiaLong = tibiaCategory === 'long' || tibiaCategory === 'slightly-long';
  const tibiaShort = tibiaCategory === 'short' || tibiaCategory === 'slightly-short';
  const mild = femurCategory.startsWith('slightly') || tibiaCategory.startsWith('slightly');

  if (femurLong && tibiaShort) {
    if (mild) {
      insights.push('Your femurs trend long and tibias trend short — this nudges the hip back with limited knee travel to compensate.');
      insights.push('Heel elevation or a slightly wider stance can help; experiment to find what feels natural.');
    } else {
      insights.push('Long femurs + short tibias is the hardest combination — the femur pushes the hip back while the tibia can\'t compensate with forward knee travel.');
      insights.push('Strongly consider: wider stance, low-bar position, and/or squat shoes.');
    }
  } else if (femurLong && tibiaLong) {
    if (mild) {
      insights.push('Femurs trend long but tibias also trend long — the extra knee travel mostly compensates, so the net effect on lean is small.');
    } else {
      insights.push('Long femurs are partially offset by long tibias — extra knee travel reduces the lean that would otherwise be needed.');
    }
  } else if (femurShort && tibiaLong) {
    if (mild) {
      insights.push('Femurs trend short with tibias trending long — depth and a fairly upright posture should come relatively easy.');
    } else {
      insights.push('Short femurs + long tibias — depth and upright posture should come naturally.');
    }
  } else if (femurShort && tibiaShort) {
    if (mild) {
      insights.push('Femurs and tibias both trend a bit short — the hip stays close but knee travel is limited, so the net effect is near-average.');
    } else {
      insights.push('Short femurs keep the hip close, but short tibias limit knee travel — net effect is roughly average difficulty.');
    }
  } else if (femurLong && !tibiaShort && !tibiaLong) {
    insights.push(`${femurCategory === 'slightly-long' ? 'Slightly long' : 'Long'} femurs with average-length tibias — expect a bit more forward lean than typical; squat shoes can help.`);
  } else if (femurShort && !tibiaShort && !tibiaLong) {
    insights.push(`${femurCategory === 'slightly-short' ? 'Slightly short' : 'Short'} femurs with average-length tibias — the hip stays relatively close to midfoot, keeping lean moderate.`);
  } else if (tibiaLong && !femurShort && !femurLong) {
    insights.push(`Average femurs with ${tibiaCategory === 'slightly-long' ? 'slightly long' : 'long'} tibias — extra knee travel helps keep the torso upright.`);
  } else if (tibiaShort && !femurShort && !femurLong) {
    insights.push(`Average femurs with ${tibiaCategory === 'slightly-short' ? 'slightly short' : 'short'} tibias — limited knee travel adds a few degrees of lean; ankle mobility work pays off here.`);
  }

  // Depth-specific cue
  if (p.legToTorso > 2.1) {
    insights.push('Long legs relative to torso — even with good ankle mobility, maintaining an upright torso below parallel will be challenging.');
  } else if (p.legToTorso < 1.7) {
    insights.push('Compact legs relative to torso — achieving full depth should be straightforward.');
  }

  return {
    proportions: p,
    femurCategory,
    tibiaCategory,
    difficultyScore,
    estimatedLeanDeg,
    estimatedParallelKneeFlexDeg: parallelFlex.kneeFlexDeg,
    estimatedParallelHipFlexDeg: parallelFlex.hipFlexProxyDeg,
    dominantPattern,
    insights,
  };
}

type AppliedZoomMap = {x: number; y: number; s: number};
type ViewportTarget = 'primary' | 'compare';
type KneeSide = 'left' | 'right';
type PosePoint = {x: number; y: number; v: number};
type PoseFrameSample = {
  time: number;
  side: KneeSide;
  kneeAngle: number;
  hipX: number;
  ankleX: number;
  extension: number;
  /** Nose Y in normalized frame coords (0 top, 1 bottom). */
  headY: number | null;
  /** Nose→lowest foot (heel/toe) vertical gap in normalized coords; scales head metrics vs “ground”. */
  headToGroundY: number | null;
  keypoints: PosePoint[];
};
type FacingDirection = 'right' | 'left';
type ExtensionDirection = 'forward' | 'behind';
type AnalysisMode = 'stride' | 'squat';
type KneeAnglePeak = {
  time: number;
  angle: number;
  direction: ExtensionDirection;
};
type KneeAngleValley = {
  time: number;
  angle: number;
  depthAngle: number;
  belowParallel: boolean;
  hipAngle: number | null;
  backAngle: number | null;
};
type PoseAngleSample = {
  time: number;
  angle: number;
  hipX: number;
  ankleX: number;
  depthAngle: number | null;
  hipAngle: number | null;
  backAngle: number | null;
};

function detectFacingFromKeypoints(
  keypoints: {x: number; y: number; v: number}[],
): FacingDirection | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const leftIds = [5, 7, 9, 11, 13, 15];
  const rightIds = [6, 8, 10, 12, 14, 16];
  let leftSum = 0, leftCount = 0;
  for (const id of leftIds) {
    const p = byId.get(id);
    if (p) { leftSum += p.v; leftCount++; }
  }
  let rightSum = 0, rightCount = 0;
  for (const id of rightIds) {
    const p = byId.get(id);
    if (p) { rightSum += p.v; rightCount++; }
  }
  if (leftCount === 0 && rightCount === 0) return null;
  const leftAvg = leftCount > 0 ? leftSum / leftCount : 0;
  const rightAvg = rightCount > 0 ? rightSum / rightCount : 0;
  if (Math.abs(leftAvg - rightAvg) < 0.05) return null;
  return leftAvg > rightAvg ? 'left' : 'right';
}

function facingDirectionToLeg(facing: FacingDirection): KneeSide {
  return facing === 'left' ? 'left' : 'right';
}

function detectKneeAngleMaxima(
  series: {time: number; angle: number; hipX: number; ankleX: number}[],
  options: {
    lowerAngle: number;
    upperAngle: number;
    facing: FacingDirection;
    minProminence?: number;
    neighborRadius?: number;
  },
): KneeAnglePeak[] {
  if (series.length < KNEE_MAXIMA_MIN_SAMPLES) return [];
  const minProm = options.minProminence ?? KNEE_MAXIMA_MIN_PROMINENCE;
  const radius = options.neighborRadius ?? KNEE_MAXIMA_NEIGHBOR_RADIUS;
  const angleOnly = series.map((s) => ({angle: s.angle}));

  const candidates: {index: number; angle: number; direction: ExtensionDirection}[] = [];
  for (let i = 1; i < series.length - 1; i += 1) {
    const prev = series[i - 1]!.angle;
    const current = series[i]!.angle;
    const next = series[i + 1]!.angle;
    const isPeak = (current >= prev && current > next) || (current > prev && current >= next);
    if (!isPeak || current < options.lowerAngle || current > options.upperAngle) continue;
    if (!isNeighborhoodMaximum(angleOnly, i, radius)) continue;
    if (peakProminence(angleOnly, i) < minProm) continue;

    const s = series[i]!;
    const delta = s.ankleX - s.hipX;
    const isForward = options.facing === 'left' ? delta < 0 : delta > 0;
    const direction: ExtensionDirection = isForward ? 'forward' : 'behind';
    candidates.push({index: i, angle: current, direction});
  }

  candidates.sort((a, b) => b.angle - a.angle);
  const kept: typeof candidates = [];
  for (const c of candidates) {
    const t = series[c.index]!.time;
    const tooClose = kept.some((k) => Math.abs(t - series[k.index]!.time) < KNEE_MAXIMA_MIN_GAP_SECONDS);
    if (!tooClose) kept.push(c);
  }

  return kept
    .sort((a, b) => a.index - b.index)
    .map((k) => ({time: series[k.index]!.time, angle: series[k.index]!.angle, direction: k.direction}));
}

function valleyProminence(
  series: {angle: number}[],
  valleyIdx: number,
): number {
  const valleyAngle = series[valleyIdx]!.angle;
  let leftMax = valleyAngle;
  for (let i = valleyIdx - 1; i >= 0; i--) {
    if (series[i]!.angle > leftMax) leftMax = series[i]!.angle;
    if (series[i]!.angle < valleyAngle) break;
  }
  let rightMax = valleyAngle;
  for (let i = valleyIdx + 1; i < series.length; i++) {
    if (series[i]!.angle > rightMax) rightMax = series[i]!.angle;
    if (series[i]!.angle < valleyAngle) break;
  }
  return Math.min(leftMax, rightMax) - valleyAngle;
}

/** Vertical prominence of a knee-extension peak (°): height above the higher of the two adjacent trough baselines. */
function peakProminence(
  series: {angle: number}[],
  peakIdx: number,
): number {
  const p = series[peakIdx]!.angle;
  let leftMin = p;
  for (let i = peakIdx - 1; i >= 0; i--) {
    if (series[i]!.angle >= p) {
      break;
    }
    if (series[i]!.angle < leftMin) {
      leftMin = series[i]!.angle;
    }
  }
  let rightMin = p;
  for (let i = peakIdx + 1; i < series.length; i++) {
    if (series[i]!.angle >= p) {
      break;
    }
    if (series[i]!.angle < rightMin) {
      rightMin = series[i]!.angle;
    }
  }
  return p - Math.max(leftMin, rightMin);
}

function isNeighborhoodMaximum(
  series: {angle: number}[],
  idx: number,
  radius: number,
): boolean {
  const p = series[idx]!.angle;
  const lo = Math.max(0, idx - radius);
  const hi = Math.min(series.length - 1, idx + radius);
  for (let j = lo; j <= hi; j++) {
    if (j !== idx && series[j]!.angle > p) {
      return false;
    }
  }
  return true;
}

function detectKneeAngleMinima(
  series: PoseAngleSample[],
  options: {
    lowerAngle: number;
    upperAngle: number;
    minProminence?: number;
  },
): KneeAngleValley[] {
  if (series.length < KNEE_MINIMA_MIN_SAMPLES) return [];
  const prominence = options.minProminence ?? KNEE_MINIMA_DEFAULT_PROMINENCE;
  const candidates: {index: number; angle: number}[] = [];
  for (let i = 1; i < series.length - 1; i += 1) {
    const prev = series[i - 1]!.angle;
    const current = series[i]!.angle;
    const next = series[i + 1]!.angle;
    const isValley = (current <= prev && current < next) || (current < prev && current <= next);
    if (!isValley || current < options.lowerAngle || current > options.upperAngle) continue;
    if (valleyProminence(series, i) < prominence) continue;
    candidates.push({index: i, angle: current});
  }

  candidates.sort((a, b) => a.angle - b.angle);
  const kept: typeof candidates = [];
  for (const c of candidates) {
    const t = series[c.index]!.time;
    const tooClose = kept.some((k) => Math.abs(t - series[k.index]!.time) < KNEE_MINIMA_MIN_GAP_SECONDS);
    if (!tooClose) kept.push(c);
  }

  return kept
    .sort((a, b) => a.index - b.index)
    .map((k) => {
      const s = series[k.index]!;
      const depth = s.depthAngle ?? 0;
      return {
        time: s.time,
        angle: s.angle,
        depthAngle: depth,
        belowParallel: depth < 0,
        hipAngle: s.hipAngle,
        backAngle: s.backAngle,
      };
    });
}

function findNearestCachedPose(
  cache: {time: number; keypoints: {x: number; y: number; v: number}[]}[],
  time: number,
): {x: number; y: number; v: number}[] | null {
  if (cache.length === 0) return null;
  if (cache.length === 1) return cache[0]!.keypoints;

  let lo = 0;
  let hi = cache.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cache[mid]!.time < time) lo = mid + 1;
    else hi = mid;
  }

  const upper = lo;
  const lower = lo > 0 ? lo - 1 : 0;
  if (upper === lower) return cache[upper]!.keypoints;

  const distLo = Math.abs(time - cache[lower]!.time);
  const distHi = Math.abs(time - cache[upper]!.time);
  return distLo <= distHi ? cache[lower]!.keypoints : cache[upper]!.keypoints;
}

const FOOT_INCL_MIN_VIS = 0.2;

/**
 * Acute angle (0–90°) between heel→toe and image horizontal. ~0° when the sole is level in the
 * frame (flat foot in side view with level camera). Uses |dx|,|dy| so left/right foot direction
 * does not flip the sign; atan2(signed dy, signed dx) was reporting ~180° for toes-behind-heel.
 */
function getFootInclinationVsHorizontalDeg(
  keypoints: PosePoint[],
  side: KneeSide,
  aspectRatio = 1,
  visibilityThreshold = FOOT_INCL_MIN_VIS,
): number | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, PosePoint>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const heel = side === 'left' ? byId.get(17) : byId.get(18);
  const toe = side === 'left' ? byId.get(19) : byId.get(20);
  if (!heel || !toe) return null;
  if (Math.min(heel.v, toe.v) < visibilityThreshold) return null;
  const dx = (toe.x - heel.x) * aspectRatio;
  const dy = toe.y - heel.y;
  if (Math.abs(dx) < 1e-9 && Math.abs(dy) < 1e-9) return null;
  return Math.atan2(Math.abs(dy), Math.abs(dx)) * (180 / Math.PI);
}

/** Nose Y in normalized image coords (0 = top); used for head height and per-stride vertical movement. */
function getHeadNormYFromKeypoints(
  keypoints: {x: number; y: number; v: number}[],
  visibilityThreshold = MIN_POSE_VISIBILITY,
): number | null {
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const i = POSE_TRACKED_IDS.indexOf(POSE_NOSE_ID);
  if (i < 0) return null;
  const p = keypoints[i];
  if (!p || p.v < visibilityThreshold) return null;
  return p.y;
}

/**
 * Vertical gap nose → lowest foot point (max of heel/toe Y) in normalized coords.
 * Ground proxy uses the same heel/toe landmarks as the toe-angle calc (side view).
 */
function getHeadToGroundNormGap(
  keypoints: {x: number; y: number; v: number}[],
  side: KneeSide,
  visibilityThreshold = MIN_POSE_VISIBILITY,
): number | null {
  const ny = getHeadNormYFromKeypoints(keypoints, visibilityThreshold);
  if (ny === null) return null;
  if (keypoints.length !== POSE_TRACKED_IDS.length) return null;
  const byId = new Map<number, {x: number; y: number; v: number}>();
  POSE_TRACKED_IDS.forEach((id, i) => {
    const p = keypoints[i];
    if (p) byId.set(id, p);
  });
  const heel = side === 'left' ? byId.get(17) : byId.get(18);
  const toe = side === 'left' ? byId.get(19) : byId.get(20);
  if (!heel || !toe) return null;
  if (Math.min(heel.v, toe.v) < visibilityThreshold) return null;
  const groundY = Math.max(heel.y, toe.y);
  const g = groundY - ny;
  return g > 1e-9 ? g : null;
}

function medianFinite(values: number[]): number | null {
  const v = values.filter((x) => Number.isFinite(x));
  if (v.length === 0) return null;
  const s = [...v].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

function nearestPoseFrameByTime(frames: PoseFrameSample[], time: number): PoseFrameSample | null {
  if (frames.length === 0) return null;
  let best = frames[0]!;
  let bestDt = Math.abs(best.time - time);
  for (let i = 1; i < frames.length; i += 1) {
    const f = frames[i]!;
    const dt = Math.abs(f.time - time);
    if (dt < bestDt) {
      best = f;
      bestDt = dt;
    }
  }
  return best;
}

/**
 * Mean head oscillation per stride as % of mean nose→foot (ground) distance within that stride.
 * Stride windows: contact-to-contact, else peak-to-peak.
 */
function computeAvgHeadMovementPerStrideVsGroundPct(
  poseFrames: PoseFrameSample[],
  contactTimes: number[],
  peakTimes: number[],
): number | null {
  if (poseFrames.length === 0) return null;
  const sorted = [...poseFrames].sort((a, b) => a.time - b.time);
  const boundaries =
    contactTimes.length >= 2
      ? [...new Set(contactTimes)].sort((a, b) => a - b)
      : peakTimes.length >= 2
        ? [...new Set(peakTimes)].sort((a, b) => a - b)
        : [];
  if (boundaries.length < 2) return null;
  const ratios: number[] = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    const t0 = boundaries[i]!;
    const t1 = boundaries[i + 1]!;
    const slice = sorted.filter((f) => f.time >= t0 && f.time <= t1);
    const headYs = slice
      .map((f) => f.headY)
      .filter((y): y is number => y !== null && y !== undefined && Number.isFinite(y));
    const gaps = slice
      .map((f) => f.headToGroundY)
      .filter((g): g is number => g !== null && g !== undefined && g > 1e-9 && Number.isFinite(g));
    if (headYs.length < 2 || gaps.length === 0) continue;
    const R = Math.max(...headYs) - Math.min(...headYs);
    const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    if (meanGap < 1e-9) continue;
    ratios.push((R / meanGap) * 100);
  }
  return ratios.length > 0 ? ratios.reduce((a, b) => a + b, 0) / ratios.length : null;
}

/**
 * Contact proxy: first clear knee flexion after extension peak (within ~120ms), not mid-stance min.
 * Uses nearest-sample angle at peak vs STRIDE_FLEX_ONSET_DEG drop; else first post-peak sample.
 */
function findContactTimeAfterExtensionPeak(
  panelSeries: PoseAngleSample[],
  peakTime: number,
): number | null {
  if (panelSeries.length < 2) return null;
  const sorted = [...panelSeries].sort((a, b) => a.time - b.time);
  const atPeak = sorted.reduce((best, s) =>
    Math.abs(s.time - peakTime) < Math.abs(best.time - peakTime) ? s : best,
  );
  const peakAngle = atPeak.angle;

  for (let i = 0; i < sorted.length; i += 1) {
    const s = sorted[i]!;
    if (s.time <= peakTime) continue;
    if (s.time > peakTime + STRIDE_CONTACT_SEARCH_SEC) break;
    if (s.angle < peakAngle - STRIDE_FLEX_ONSET_DEG) {
      return s.time;
    }
  }

  const firstAfter = sorted.find((s) => s.time > peakTime);
  if (firstAfter && firstAfter.time <= peakTime + STRIDE_CONTACT_SEARCH_SEC) {
    return firstAfter.time;
  }

  const slice = sorted.filter(
    (s) => s.time > peakTime && s.time <= peakTime + STRIDE_CONTACT_SEARCH_SEC,
  );
  if (slice.length >= 3) {
    for (let i = 1; i < slice.length - 1; i += 1) {
      const prev = slice[i - 1]!.angle;
      const cur = slice[i]!.angle;
      const next = slice[i + 1]!.angle;
      if (cur <= prev && cur < next) {
        return slice[i]!.time;
      }
    }
  }
  if (slice.length > 0) return slice[0]!.time;
  return peakTime + 1 / 30;
}

/** Media/content coordinates → overlay (viewport) pixels — identity when not zoomed. */
function contentToOverlay(px: number, py: number, applied: AppliedZoomMap | null) {
  if (!applied) return {x: px, y: py};
  return {x: (px - applied.x) * applied.s, y: (py - applied.y) * applied.s};
}

/** Overlay (viewport) pixels → media/content coordinates. */
function overlayToContent(ox: number, oy: number, applied: AppliedZoomMap | null) {
  if (!applied) return {x: ox, y: oy};
  return {x: ox / applied.s + applied.x, y: oy / applied.s + applied.y};
}

export default function App() {
  const { accentId } = useTheme();
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [compareVideoSrc, setCompareVideoSrc] = useState<string | null>(null);
  const [compareImageSrc, setCompareImageSrc] = useState<string | null>(null);
  const [showAnalysis, setShowAnalysis] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const compareVideoRef = useRef<HTMLVideoElement>(null);
  const compareImageRef = useRef<HTMLImageElement>(null);
  const compareCanvasRef = useRef<HTMLCanvasElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaWrapRef = useRef<HTMLDivElement>(null);
  const compareMediaWrapRef = useRef<HTMLDivElement>(null);
  /** Gallery: images + videos from files — never use `capture` on this input. */
  const galleryPickInputRef = useRef<HTMLInputElement>(null);
  /** Native still-photo fallback only: `image/*` + `capture="environment"` via JS. */
  const imagePickInputRef = useRef<HTMLInputElement>(null);
  /** Native video-capture fallback only: `video/*` + `capture="environment"` via JS. */
  const videoPickInputRef = useRef<HTMLInputElement>(null);
  const comparePickInputRef = useRef<HTMLInputElement>(null);
  const cameraPreviewRef = useRef<HTMLVideoElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const videoRecordPreviewRef = useRef<HTMLVideoElement>(null);
  const videoRecordStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const videoSaveOnStopRef = useRef(false);
  const [cameraModalOpen, setCameraModalOpen] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [videoRecordModalOpen, setVideoRecordModalOpen] = useState(false);
  const [videoRecordError, setVideoRecordError] = useState<string | null>(null);
  const [isRecordingVideo, setIsRecordingVideo] = useState(false);
  const [mediaLayoutVersion, setMediaLayoutVersion] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  type Measurement = {
    points: { x: number; y: number }[];
    kind: 'angle' | 'line';
    angle: number | null;
    length: number | null;
    viewport: ViewportTarget;
  };
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [activeMeasurementIdx, setActiveMeasurementIdx] = useState<number | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  const points = activeMeasurementIdx !== null ? (measurements[activeMeasurementIdx]?.points ?? []) : [];
  const angle = activeMeasurementIdx !== null ? (measurements[activeMeasurementIdx]?.angle ?? null) : null;
  const setPoints = (updater: { x: number; y: number }[] | ((prev: { x: number; y: number }[]) => { x: number; y: number }[])) => {
    if (activeMeasurementIdx === null) return;
    setMeasurements(prev => {
      const copy = [...prev];
      const m = copy[activeMeasurementIdx];
      if (!m) return prev;
      const newPts = typeof updater === 'function' ? updater(m.points) : updater;
      copy[activeMeasurementIdx] = { ...m, points: newPts };
      return copy;
    });
  };
  const setAngle = (v: number | null) => {
    if (activeMeasurementIdx === null) return;
    setMeasurements(prev => {
      const copy = [...prev];
      const m = copy[activeMeasurementIdx];
      if (!m) return prev;
      copy[activeMeasurementIdx] = { ...m, angle: v };
      return copy;
    });
  };
  const setLength = (v: number | null) => {
    if (activeMeasurementIdx === null) return;
    setMeasurements(prev => {
      const copy = [...prev];
      const m = copy[activeMeasurementIdx];
      if (!m) return prev;
      copy[activeMeasurementIdx] = { ...m, length: v };
      return copy;
    });
  };
  const [poseEnabled, setPoseEnabled] = useState(false);
  const [poseKeypoints, setPoseKeypoints] = useState<{x: number; y: number; v: number}[]>([]);
  const [comparePoseKeypoints, setComparePoseKeypoints] = useState<{x: number; y: number; v: number}[]>([]);
  const [currentKneeAngle, setCurrentKneeAngle] = useState<number | null>(null);
  const [currentLiveDepthAngle, setCurrentLiveDepthAngle] = useState<number | null>(null);
  const [currentLiveHipAngle, setCurrentLiveHipAngle] = useState<number | null>(null);
  const [currentLiveBackAngle, setCurrentLiveBackAngle] = useState<number | null>(null);
  const [kneeAngleSeries, setKneeAngleSeries] = useState<PoseAngleSample[]>([]);
  const [compareCurrentKneeAngle, setCompareCurrentKneeAngle] = useState<number | null>(null);
  const [compareCurrentLiveDepthAngle, setCompareCurrentLiveDepthAngle] = useState<number | null>(null);
  const [compareCurrentLiveHipAngle, setCompareCurrentLiveHipAngle] = useState<number | null>(null);
  const [compareCurrentLiveBackAngle, setCompareCurrentLiveBackAngle] = useState<number | null>(null);
  const [compareKneeAngleSeries, setCompareKneeAngleSeries] = useState<PoseAngleSample[]>([]);
  const [posePointSeries, setPosePointSeries] = useState<PoseFrameSample[]>([]);
  const [comparePosePointSeries, setComparePosePointSeries] = useState<PoseFrameSample[]>([]);
  const [bodyProportions, setBodyProportions] = useState<ProportionProfile | null>(null);
  const [liveBodyProportions, setLiveBodyProportions] = useState<BodyProportions | null>(null);
  const [compareBodyProportions, setCompareBodyProportions] = useState<ProportionProfile | null>(null);
  const [compareLiveBodyProportions, setCompareLiveBodyProportions] = useState<BodyProportions | null>(null);
  const [kneeTrackingSide, setKneeTrackingSide] = useState<KneeSide | null>(null);
  const [compareKneeTrackingSide, setCompareKneeTrackingSide] = useState<KneeSide | null>(null);
  const [isPoseAnalyzing, setIsPoseAnalyzing] = useState(false);
  const [isKneeGraphLocked, setIsKneeGraphLocked] = useState(false);
  const [graphAnalysisRequested, setGraphAnalysisRequested] = useState(false);
  const [graphDomainTime, setGraphDomainTime] = useState(0);
  const [angleLowerBound, setAngleLowerBound] = useState(100);
  const [angleUpperBound, setAngleUpperBound] = useState(180);
  const [draggingBound, setDraggingBound] = useState<'lower' | 'upper' | null>(null);
  const [draggingPlayhead, setDraggingPlayhead] = useState(false);
  const wasPlayingBeforeScrubRef = useRef(false);
  const graphSvgRef = useRef<SVGSVGElement | null>(null);
  const [draggingComparePlayhead, setDraggingComparePlayhead] = useState(false);
  const wasPlayingBeforeCompareScrubRef = useRef(false);
  const compareGraphSvgRef = useRef<SVGSVGElement | null>(null);
  const [showAngleGuide, setShowAngleGuide] = useState(false);
  const [showBiomechanics, setShowBiomechanics] = useState(false);
  const [extensionFilter, setExtensionFilter] = useState<ExtensionDirection>('forward');
  const [primaryFacingDirection, setPrimaryFacingDirection] = useState<FacingDirection>('right');
  const [compareFacingDirection, setCompareFacingDirection] = useState<FacingDirection>('right');
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('stride');
  const poseDetectorRef = useRef<poseDetection.PoseDetector | null>(null);
  /** Reused frame buffer so BlazePose sees correct canvas dimensions for video. */
  const poseFrameCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const poseRafRef = useRef<number | null>(null);
  const bgVideoRef = useRef<HTMLVideoElement | null>(null);
  const [poseStatus, setPoseStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const poseCacheRef = useRef<{time: number; keypoints: {x: number; y: number; v: number}[]}[]>([]);
  const comparePoseCacheRef = useRef<{time: number; keypoints: {x: number; y: number; v: number}[]}[]>([]);
  const [analysisProgress, setAnalysisProgress] = useState<number | null>(null);
  const analysisAbortRef = useRef(false);
  const analysisGenRef = useRef(0);
  const [compareMediaLayoutVersion, setCompareMediaLayoutVersion] = useState(0);
  const compareBgVideoRef = useRef<HTMLVideoElement | null>(null);

  const mapPoseNormToCanvasOverlayPx = (
    xn: number,
    yn: number,
    media: HTMLVideoElement | HTMLImageElement | null,
    canvas: HTMLCanvasElement | null,
  ) => {
    if (!media || !canvas) return {x: xn * (canvas?.width ?? 1), y: yn * (canvas?.height ?? 1)};

    const canvasRect = canvas.getBoundingClientRect();
    const mediaRect = media.getBoundingClientRect();
    if (canvasRect.width <= 0 || canvasRect.height <= 0 || mediaRect.width <= 0 || mediaRect.height <= 0) {
      return {x: xn * canvas.width, y: yn * canvas.height};
    }

    const sx = canvas.width / canvasRect.width;
    const sy = canvas.height / canvasRect.height;

    // Pose landmarks are normalized in the video/image pixel space, so scale them into
    // the actual displayed media bounding box (including any transforms/letterboxing).
    return {
      x: (mediaRect.left - canvasRect.left) * sx + xn * mediaRect.width * sx,
      y: (mediaRect.top - canvasRect.top) * sy + yn * mediaRect.height * sy,
    };
  };

  /** `angle` = three-point angle, `line` = two-point distance, `pan` = drag viewport while zoomed */
  type AnalysisTool = 'angle' | 'line' | 'pan' | null;
  const [activeTool, setActiveTool] = useState<AnalysisTool>(null);
  const [activeViewport, setActiveViewport] = useState<ViewportTarget>('primary');
  const [appliedZoom, setAppliedZoom] = useState<{x: number; y: number; s: number} | null>(null);
  const [compareAppliedZoom, setCompareAppliedZoom] = useState<{x: number; y: number; s: number} | null>(null);
  const [isPanning, setIsPanning] = useState(false);
  const [isComparePanning, setIsComparePanning] = useState(false);
  const activePointersRef = useRef<Map<number, {clientX: number; clientY: number}>>(new Map());
  const compareActivePointersRef = useRef<Map<number, {clientX: number; clientY: number}>>(new Map());
  const panDragRef = useRef<{
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startX: number;
    startY: number;
  } | null>(null);
  const comparePanDragRef = useRef<{
    pointerId: number;
    startClientX: number;
    startClientY: number;
    startX: number;
    startY: number;
  } | null>(null);
  const pinchRef = useRef<{
    initialDistance: number;
    initialScale: number;
    anchorX: number;
    anchorY: number;
    initialX: number;
    initialY: number;
  } | null>(null);
  const comparePinchRef = useRef<{
    initialDistance: number;
    initialScale: number;
    anchorX: number;
    anchorY: number;
    initialX: number;
    initialY: number;
  } | null>(null);

  const resetZoomAll = () => {
    setAppliedZoom(null);
    setCompareAppliedZoom(null);
    setActiveTool((prev) => (prev === 'pan' ? null : prev));
    setIsPanning(false);
    setIsComparePanning(false);
    panDragRef.current = null;
    comparePanDragRef.current = null;
    pinchRef.current = null;
    comparePinchRef.current = null;
    activePointersRef.current.clear();
    compareActivePointersRef.current.clear();
  };

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setMeasurements([]);
      setActiveMeasurementIdx(null);
      setPoseKeypoints([]);
      setComparePoseKeypoints([]);
      resetZoomAll();
      if (file.type.startsWith('video/')) {
        setVideoSrc(URL.createObjectURL(file));
        setImageSrc(null);
      } else if (file.type.startsWith('image/')) {
        setImageSrc(URL.createObjectURL(file));
        setVideoSrc(null);
      }
    }
    event.target.value = '';
  };

  const handleCompareFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('video/')) {
        setCompareVideoSrc(URL.createObjectURL(file));
        setCompareImageSrc(null);
      } else if (file.type.startsWith('image/')) {
        setCompareImageSrc(URL.createObjectURL(file));
        setCompareVideoSrc(null);
      }
      setCompareAppliedZoom(null);
      setIsComparePanning(false);
      setComparePoseKeypoints([]);
    }
    event.target.value = '';
  };

  const stopCameraStream = () => {
    cameraStreamRef.current?.getTracks().forEach((t) => t.stop());
    cameraStreamRef.current = null;
    const v = cameraPreviewRef.current;
    if (v) v.srcObject = null;
  };

  const closeCameraModal = () => {
    stopCameraStream();
    setCameraModalOpen(false);
    setCameraError(null);
  };

  const openNativeCameraFilePicker = () => {
    const el = imagePickInputRef.current;
    if (!el) return;
    el.setAttribute('capture', 'environment');
    el.click();
  };

  const openGalleryPicker = () => {
    const el = galleryPickInputRef.current;
    if (!el) return;
    el.removeAttribute('capture');
    el.click();
  };

  const openComparePicker = () => {
    const el = comparePickInputRef.current;
    if (!el) return;
    el.removeAttribute('capture');
    el.click();
  };

  const openNativeVideoCapturePicker = () => {
    const el = videoPickInputRef.current;
    if (!el) return;
    el.setAttribute('capture', 'environment');
    el.click();
  };

  const closeVideoRecordModal = () => {
    videoSaveOnStopRef.current = false;
    const rec = mediaRecorderRef.current;
    if (rec && rec.state !== 'inactive') {
      rec.onstop = null;
      try {
        rec.stop();
      } catch {
        /* ignore */
      }
    }
    mediaRecorderRef.current = null;
    recordedChunksRef.current = [];
    videoRecordStreamRef.current?.getTracks().forEach((t) => t.stop());
    videoRecordStreamRef.current = null;
    const v = videoRecordPreviewRef.current;
    if (v) v.srcObject = null;
    setIsRecordingVideo(false);
    setVideoRecordModalOpen(false);
    setVideoRecordError(null);
  };

  /** Live video + optional mic for MediaRecorder (device video). */
  const getVideoStreamForRecord = async (): Promise<MediaStream> => {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('getUserMedia unavailable');
    }
    const attempts: MediaStreamConstraints[] = [
      {video: {facingMode: {ideal: 'environment'}}, audio: true},
      {video: true, audio: true},
      {video: {facingMode: {ideal: 'environment'}}, audio: false},
      {video: true, audio: false},
    ];
    let last: unknown;
    for (const constraints of attempts) {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints);
      } catch (e) {
        last = e;
      }
    }
    throw last;
  };

  const openDeviceVideoRecord = async () => {
    setVideoRecordError(null);
    if (typeof MediaRecorder === 'undefined') {
      openNativeVideoCapturePicker();
      return;
    }
    try {
      const stream = await getVideoStreamForRecord();
      videoRecordStreamRef.current?.getTracks().forEach((t) => t.stop());
      videoRecordStreamRef.current = stream;
      setVideoRecordModalOpen(true);
    } catch {
      openNativeVideoCapturePicker();
    }
  };

  useEffect(() => {
    if (!videoRecordModalOpen) return;
    const video = videoRecordPreviewRef.current;
    const stream = videoRecordStreamRef.current;
    if (!video || !stream) return;
    video.srcObject = stream;
    void video.play().catch(() => {
      setVideoRecordError('Could not show camera preview. Try gallery or your device capture.');
    });
    return () => {
      video.srcObject = null;
    };
  }, [videoRecordModalOpen]);

  const startVideoRecording = () => {
    const stream = videoRecordStreamRef.current;
    if (!stream) return;
    const mimeType = pickVideoRecorderMimeType();
    try {
      const rec = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
      recordedChunksRef.current = [];
      videoSaveOnStopRef.current = false;
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      rec.onstop = () => {
        const save = videoSaveOnStopRef.current;
        videoSaveOnStopRef.current = false;
        const chunks = recordedChunksRef.current;
        recordedChunksRef.current = [];
        mediaRecorderRef.current = null;
        stream.getTracks().forEach((t) => t.stop());
        videoRecordStreamRef.current = null;
        const pv = videoRecordPreviewRef.current;
        if (pv) pv.srcObject = null;
        setIsRecordingVideo(false);
        if (!save) {
          return;
        }
        if (chunks.length === 0) {
          setVideoRecordError('No video data recorded.');
          return;
        }
        const blobType = rec.mimeType || mimeType || 'video/webm';
        const blob = new Blob(chunks, {type: blobType});
        const ext = blobType.includes('mp4') ? 'mp4' : 'webm';
        const file = new File([blob], `record-${Date.now()}.${ext}`, {type: blobType});
        setMeasurements([]);
        setActiveMeasurementIdx(null);
        setImageSrc(null);
        setVideoSrc(URL.createObjectURL(file));
        setVideoRecordModalOpen(false);
        setVideoRecordError(null);
      };
      mediaRecorderRef.current = rec;
      rec.start(250);
      setIsRecordingVideo(true);
    } catch {
      videoRecordStreamRef.current?.getTracks().forEach((t) => t.stop());
      videoRecordStreamRef.current = null;
      const pv = videoRecordPreviewRef.current;
      if (pv) pv.srcObject = null;
      setVideoRecordModalOpen(false);
      setIsRecordingVideo(false);
      setVideoRecordError('Recording not supported here. Try gallery or native capture.');
      openNativeVideoCapturePicker();
    }
  };

  const stopVideoRecordingAndSave = () => {
    const rec = mediaRecorderRef.current;
    if (!rec || rec.state === 'inactive') return;
    videoSaveOnStopRef.current = true;
    rec.stop();
  };

  /** getUserMedia: prefer rear camera, then any video device; then file input with capture. */
  const getCameraStream = async (): Promise<MediaStream> => {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('getUserMedia unavailable');
    }
    try {
      return await navigator.mediaDevices.getUserMedia({
        video: {facingMode: {ideal: 'environment'}},
        audio: false,
      });
    } catch {
      return await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
    }
  };

  const openDeviceCamera = async () => {
    setCameraError(null);

    try {
      const stream = await getCameraStream();
      stopCameraStream();
      cameraStreamRef.current = stream;
      setCameraModalOpen(true);
    } catch {
      openNativeCameraFilePicker();
    }
  };

  useEffect(() => {
    if (!cameraModalOpen) return;
    const video = cameraPreviewRef.current;
    const stream = cameraStreamRef.current;
    if (!video || !stream) return;
    video.srcObject = stream;
    void video.play().catch(() => {
      setCameraError('Playback failed. Try capturing from the file picker instead.');
    });
    return () => {
      video.srcObject = null;
    };
  }, [cameraModalOpen]);

  useEffect(() => {
    return () => {
      stopCameraStream();
      videoSaveOnStopRef.current = false;
      const rec = mediaRecorderRef.current;
      if (rec && rec.state !== 'inactive') {
        rec.onstop = null;
        try {
          rec.stop();
        } catch {
          /* ignore */
        }
      }
      mediaRecorderRef.current = null;
      videoRecordStreamRef.current?.getTracks().forEach((t) => t.stop());
      videoRecordStreamRef.current = null;
    };
  }, []);

  const capturePhotoFromCamera = () => {
    const video = cameraPreviewRef.current;
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      setCameraError('Camera not ready yet. Wait a moment and try again.');
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(
      (blob) => {
        if (!blob) {
          setCameraError('Could not capture image.');
          return;
        }
        const file = new File([blob], `capture-${Date.now()}.jpg`, {
          type: 'image/jpeg',
        });
        stopCameraStream();
        setCameraModalOpen(false);
        setCameraError(null);
        setMeasurements([]);
        setActiveMeasurementIdx(null);
        setVideoSrc(null);
        setImageSrc(URL.createObjectURL(file));
      },
      'image/jpeg',
      0.92,
    );
  };

  const stepPrimaryFrame = (direction: number) => {
    if (!videoRef.current) return;
    const v = videoRef.current;
    v.currentTime = Math.max(0, v.currentTime + direction * FRAME_STEP_SECONDS);
  };

  const stepCompareFrame = (direction: number) => {
    if (!compareVideoRef.current) return;
    const v = compareVideoRef.current;
    v.currentTime = Math.max(0, v.currentTime + direction * FRAME_STEP_SECONDS);
  };

  const stepBothFrames = (direction: number) => {
    if (videoRef.current) stepPrimaryFrame(direction);
    if (compareVideoRef.current && compareVideoSrc) {
      stepCompareFrame(direction);
    }
  };

  /** Pause/play in layout phase + parallel play() so both videos begin decoding together. */
  useLayoutEffect(() => {
    const primary = videoRef.current;
    const compare = compareVideoRef.current;

    if (!isPlaying) {
      primary?.pause();
      if (compareVideoSrc) compare?.pause();
      return;
    }

    if (!primary) return;

    if (compareVideoSrc && compare) {
      void Promise.all([primary.play(), compare.play()]).catch(() => {});
    } else {
      void primary.play().catch(() => {});
    }
  }, [isPlaying, compareVideoSrc]);

  useEffect(() => {
    const wrap = mediaWrapRef.current;
    const target = videoRef.current ?? imageRef.current;
    if (!wrap || !target) return;

    const ro = new ResizeObserver(() => setMediaLayoutVersion((v) => v + 1));
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [videoSrc, imageSrc]);

  useEffect(() => {
    const wrap = compareMediaWrapRef.current;
    const target = compareVideoRef.current ?? compareImageRef.current;
    if (!wrap || !target) return;

    const ro = new ResizeObserver(() => setCompareMediaLayoutVersion((v) => v + 1));
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [compareVideoSrc, compareImageSrc]);

  useEffect(() => {
    return () => {
      if (poseRafRef.current !== null) {
        cancelAnimationFrame(poseRafRef.current);
        poseRafRef.current = null;
      }
      analysisAbortRef.current = true;
      poseDetectorRef.current?.dispose();
      poseDetectorRef.current = null;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (scrubRafRef.current !== null) {
        cancelAnimationFrame(scrubRafRef.current);
        scrubRafRef.current = null;
      }
      if (compareScrubRafRef.current !== null) {
        cancelAnimationFrame(compareScrubRafRef.current);
        compareScrubRafRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!poseEnabled || (!videoSrc && !imageSrc && !compareVideoSrc && !compareImageSrc)) {
      setPoseKeypoints([]);
      setComparePoseKeypoints([]);
      if (poseRafRef.current !== null) {
        cancelAnimationFrame(poseRafRef.current);
        poseRafRef.current = null;
      }
      return;
    }

    let cancelled = false;
    let lastPrimaryVideoTime = -1;
    let lastCompareVideoTime = -1;
    let lastPrimaryInferenceTs = 0;
    let lastCompareInferenceTs = 0;
    const minInferenceIntervalMs = 1000 / POSE_INFERENCE_HZ;

    const ensureDetector = async () => {
      if (poseDetectorRef.current) return poseDetectorRef.current;
      setPoseStatus('loading');
      if (tf.getBackend() !== 'webgl') {
        try {
          await tf.setBackend('webgl');
        } catch {
          // Fallback to whichever backend tfjs can initialize.
        }
      }
      await tf.ready();
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.BlazePose,
        {
          runtime: 'tfjs',
          modelType: 'full',
          enableSmoothing: true,
        },
      );
      poseDetectorRef.current = detector;
      setPoseStatus('ready');
      return detector;
    };

    const updatePoseFromResult = (
      poses: poseDetection.Pose[] | null | undefined,
      source: HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | null,
      setter: (next: {x: number; y: number; v: number}[]) => void,
      autoDetectFacing = false,
      onDetectFacing?: (facing: FacingDirection) => void,
      onDetectSide?: (side: KneeSide) => void,
    ) => {
      const keypoints = poses?.[0]?.keypoints;
      const sourceWidth =
        source instanceof HTMLCanvasElement
          ? source.width
          : source instanceof HTMLVideoElement
            ? source.videoWidth
            : source instanceof HTMLImageElement
              ? source.naturalWidth
              : 0;
      const sourceHeight =
        source instanceof HTMLCanvasElement
          ? source.height
          : source instanceof HTMLVideoElement
            ? source.videoHeight
            : source instanceof HTMLImageElement
              ? source.naturalHeight
              : 0;
      if (!keypoints || keypoints.length === 0 || sourceWidth <= 0 || sourceHeight <= 0) {
        setter([]);
        return;
      }
      const selected = keypointsFromBlazePoseOutput(keypoints, sourceWidth, sourceHeight);
      setter(selected);
      if (autoDetectFacing) {
        const detected = detectFacingFromKeypoints(selected);
        if (detected) {
          onDetectFacing?.(detected);
          onDetectSide?.(facingDirectionToLeg(detected));
        }
      }
    };

    const run = async () => {
      try {
        const detector = await ensureDetector();
        if (cancelled) return;

        const primaryVideo = videoRef.current;
        const compareVideo = compareVideoRef.current;
        const primaryImage = imageRef.current;
        const compareImage = compareImageRef.current;
        const hasAnyVideo = (!!videoSrc && !!primaryVideo) || (!!compareVideoSrc && !!compareVideo);

        if (!hasAnyVideo) {
          if (imageSrc && primaryImage) {
            const result = await detector.estimatePoses(primaryImage, {flipHorizontal: false});
            if (!cancelled) {
              updatePoseFromResult(
                result,
                primaryImage,
                setPoseKeypoints,
                true,
                setPrimaryFacingDirection,
                setKneeTrackingSide,
              );
            }
          }
          if (compareImageSrc && compareImage) {
            const result = await detector.estimatePoses(compareImage, {flipHorizontal: false});
            if (!cancelled) {
              updatePoseFromResult(
                result,
                compareImage,
                setComparePoseKeypoints,
                true,
                setCompareFacingDirection,
                setCompareKneeTrackingSide,
              );
            }
          }
          return;
        }

        if (imageSrc && primaryImage) {
          // Primary is an image: detect once and keep it.
          const result = await detector.estimatePoses(primaryImage, {flipHorizontal: false});
          if (!cancelled) {
            updatePoseFromResult(
              result,
              primaryImage,
              setPoseKeypoints,
              true,
              setPrimaryFacingDirection,
              setKneeTrackingSide,
            );
          }
        }
        if (compareImageSrc && compareImage) {
          const result = await detector.estimatePoses(compareImage, {flipHorizontal: false});
          if (!cancelled) {
            updatePoseFromResult(
              result,
              compareImage,
              setComparePoseKeypoints,
              true,
              setCompareFacingDirection,
              setCompareKneeTrackingSide,
            );
          }
        }

        const tick = async () => {
          if (cancelled) return;
          const now = performance.now();
          if (!poseFrameCanvasRef.current) {
            poseFrameCanvasRef.current = document.createElement('canvas');
          }
          const frameCanvas = poseFrameCanvasRef.current;

          if (videoSrc && primaryVideo) {
            const v = primaryVideo;
            if (
              v.readyState >= 2 &&
              (v.currentTime !== lastPrimaryVideoTime || !v.paused) &&
              now - lastPrimaryInferenceTs >= minInferenceIntervalMs
            ) {
              lastPrimaryVideoTime = v.currentTime;
              lastPrimaryInferenceTs = now;
              const blazeposeInput = blazeposeVideoFrameCanvas(v, frameCanvas);
              if (blazeposeInput) {
                const result = await detector.estimatePoses(blazeposeInput, {flipHorizontal: false});
                if (!cancelled) {
                  updatePoseFromResult(
                    result,
                    blazeposeInput,
                    setPoseKeypoints,
                    true,
                    setPrimaryFacingDirection,
                    setKneeTrackingSide,
                  );
                }
              }
            }
          }

          if (compareVideoSrc && compareVideo) {
            const v = compareVideo;
            if (
              v.readyState >= 2 &&
              (v.currentTime !== lastCompareVideoTime || !v.paused) &&
              now - lastCompareInferenceTs >= minInferenceIntervalMs
            ) {
              lastCompareVideoTime = v.currentTime;
              lastCompareInferenceTs = now;
              const blazeposeInput = blazeposeVideoFrameCanvas(v, frameCanvas);
              if (blazeposeInput) {
                const result = await detector.estimatePoses(blazeposeInput, {flipHorizontal: false});
                if (!cancelled) {
                  updatePoseFromResult(
                    result,
                    blazeposeInput,
                    setComparePoseKeypoints,
                    true,
                    setCompareFacingDirection,
                    setCompareKneeTrackingSide,
                  );
                }
              }
            }
          }

          poseRafRef.current = requestAnimationFrame(tick);
        };

        tick();
      } catch (e) {
        console.error('Pose detector failed:', e);
        if (!cancelled) setPoseStatus('error');
      }
    };

    void run();

    return () => {
      cancelled = true;
      if (poseRafRef.current !== null) {
        cancelAnimationFrame(poseRafRef.current);
        poseRafRef.current = null;
      }
    };
  }, [poseEnabled, videoSrc, imageSrc, compareVideoSrc, compareImageSrc]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const image = imageRef.current;
    if (!canvas || (!video && !image)) return;

    const wrap = canvas.parentElement;
    if (!wrap) return;

    const draw = () => {
      const vw = wrap.clientWidth;
      const vh = wrap.clientHeight;
      canvas.width = Math.max(1, Math.round(vw));
      canvas.height = Math.max(1, Math.round(vh));

      const zMap: AppliedZoomMap | null = appliedZoom
        ? {x: appliedZoom.x, y: appliedZoom.y, s: appliedZoom.s}
        : null;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const accent =
        getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim() ||
        '#ff8800';

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = accent;
      ctx.fillStyle = accent;
      ctx.lineWidth = 3;

      const primaryMeasurements = measurements.filter(m => m.viewport === 'primary');
      for (const m of primaryMeasurements) {
        if (m.points.length > 0) {
          ctx.fillStyle = accent;
          m.points.forEach(p => {
            const o = contentToOverlay(p.x, p.y, zMap);
            ctx.beginPath();
            ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }

      const overlayTime = videoRef.current?.currentTime ?? 0;
      let overlayKps = poseCacheRef.current.length > 0
        ? findNearestCachedPose(poseCacheRef.current, overlayTime) ?? poseKeypoints
        : poseKeypoints;

      if (poseEnabled && overlayKps.length === POSE_TRACKED_IDS.length) {
        const byId = new Map<number, {x: number; y: number; v: number}>();
        POSE_TRACKED_IDS.forEach((id, i) => {
          const p = overlayKps[i];
          if (p) byId.set(id, p);
        });

        ctx.save();
        const trackedIds = kneeTrackingSide === 'left' ? POSE_LEFT_SIDE_IDS : POSE_RIGHT_SIDE_IDS;
        const isTrackedConnection = (aId: number, bId: number) =>
          trackedIds.has(aId) && trackedIds.has(bId);
        const isArmConnection = (aId: number, bId: number) =>
          POSE_ARM_IDS.has(aId) || POSE_ARM_IDS.has(bId);
        const isHeadConnection = (aId: number, bId: number) =>
          aId === POSE_NOSE_ID || bId === POSE_NOSE_ID;

        for (const [aId, bId] of POSE_BODY_CONNECTIONS) {
          const a = byId.get(aId);
          const b = byId.get(bId);
          if (!a || !b || a.v < 0.25 || b.v < 0.25) continue;
          const head = isHeadConnection(aId, bId);
          const arm = isArmConnection(aId, bId);
          const tracked = !arm && !head && isTrackedConnection(aId, bId);
          ctx.strokeStyle = head ? 'rgba(0,210,255,0.15)' : arm ? 'rgba(0,210,255,0.15)' : tracked ? '#00d2ff' : 'rgba(0,210,255,0.25)';
          ctx.lineWidth = head ? 1 : arm ? 1 : tracked ? 3 : 1.5;
          const aPx = mapPoseNormToCanvasOverlayPx(a.x, a.y, videoRef.current ?? imageRef.current, canvasRef.current);
          const bPx = mapPoseNormToCanvasOverlayPx(b.x, b.y, videoRef.current ?? imageRef.current, canvasRef.current);
          ctx.beginPath();
          ctx.moveTo(aPx.x, aPx.y);
          ctx.lineTo(bPx.x, bPx.y);
          ctx.stroke();
        }
        POSE_TRACKED_IDS.forEach((id, i) => {
          const p = overlayKps[i];
          if (!p || p.v < 0.25) return;
          const arm = POSE_ARM_IDS.has(id);
          const nose = id === POSE_NOSE_ID;
          const tracked = !arm && !nose && trackedIds.has(id);
          ctx.fillStyle = nose ? 'rgba(0,210,255,0.4)' : arm ? 'rgba(0,210,255,0.15)' : tracked ? '#00d2ff' : 'rgba(0,210,255,0.25)';
          const o = mapPoseNormToCanvasOverlayPx(p.x, p.y, videoRef.current ?? imageRef.current, canvasRef.current);
          ctx.beginPath();
          ctx.arc(o.x, o.y, nose ? 3 : arm ? 2.5 : tracked ? 5 : 3.5, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
      }

      for (const m of primaryMeasurements) {
        if (m.kind === 'line' && m.points.length >= 2) {
          ctx.strokeStyle = accent;
          ctx.lineWidth = 3;
          const o0 = contentToOverlay(m.points[0].x, m.points[0].y, zMap);
          const o1 = contentToOverlay(m.points[1].x, m.points[1].y, zMap);
          ctx.beginPath();
          ctx.moveTo(o0.x, o0.y);
          ctx.lineTo(o1.x, o1.y);
          ctx.stroke();
          if (m.length !== null) {
            const midX = (o0.x + o1.x) / 2;
            const midY = (o0.y + o1.y) / 2;
            ctx.font = '20px Space Grotesk';
            ctx.fillStyle = accent;
            ctx.fillText(`${m.length.toFixed(0)} px`, midX + 8, midY - 8);
          }
        } else if (m.kind === 'angle' && m.points.length >= 2) {
          ctx.strokeStyle = accent;
          ctx.lineWidth = 3;
          const o0 = contentToOverlay(m.points[0].x, m.points[0].y, zMap);
          const o1 = contentToOverlay(m.points[1].x, m.points[1].y, zMap);
          ctx.beginPath();
          ctx.moveTo(o0.x, o0.y);
          ctx.lineTo(o1.x, o1.y);
          if (m.points.length === 3) {
            const o2 = contentToOverlay(m.points[2].x, m.points[2].y, zMap);
            ctx.lineTo(o2.x, o2.y);

            ctx.font = '20px Space Grotesk';
            ctx.fillStyle = accent;
            ctx.fillText(`${m.angle?.toFixed(1)}°`, o1.x + 10, o1.y - 10);
          }
          ctx.stroke();
        }
      }
    };

    draw();

    // RAF loop: redraw overlay at display refresh rate so the skeleton
    // tracks video.currentTime directly from the DOM instead of waiting
    // for the ~4 Hz timeupdate → React state cycle.
    if (videoSrc && poseEnabled) {
      let rafId: number | null = null;
      let lastDrawnTime = -1;
      const loop = () => {
        const vt = videoRef.current?.currentTime ?? -1;
        if (vt !== lastDrawnTime) {
          lastDrawnTime = vt;
          draw();
        }
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
      return () => {
        if (rafId !== null) cancelAnimationFrame(rafId);
      };
    }
  }, [measurements, videoSrc, imageSrc, mediaLayoutVersion, appliedZoom, poseEnabled, poseKeypoints, accentId, kneeTrackingSide, currentTime]);

  // Compare panel pose overlay (independent from primary zoom/pan)
  useEffect(() => {
    const canvas = compareCanvasRef.current;
    const video = compareVideoRef.current;
    const image = compareImageRef.current;
    if (!canvas || (!video && !image)) return;

    const wrap = canvas.parentElement;
    if (!wrap) return;

    const draw = () => {
      const vw = wrap.clientWidth;
      const vh = wrap.clientHeight;
      canvas.width = Math.max(1, Math.round(vw));
      canvas.height = Math.max(1, Math.round(vh));

      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const accent =
        getComputedStyle(document.documentElement).getPropertyValue('--color-accent').trim() ||
        '#ff8800';

      const cZMap: AppliedZoomMap | null = compareAppliedZoom
        ? {x: compareAppliedZoom.x, y: compareAppliedZoom.y, s: compareAppliedZoom.s}
        : null;

      const compareMeasurements = measurements.filter(m => m.viewport === 'compare');
      for (const m of compareMeasurements) {
        if (m.points.length > 0) {
          ctx.fillStyle = accent;
          m.points.forEach(p => {
            const o = contentToOverlay(p.x, p.y, cZMap);
            ctx.beginPath();
            ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }

      if (poseEnabled) {
        const overlayTime = compareVideoRef.current?.currentTime ?? 0;
        const overlayKps = comparePoseCacheRef.current.length > 0
          ? findNearestCachedPose(comparePoseCacheRef.current, overlayTime) ?? comparePoseKeypoints
          : comparePoseKeypoints;
        if (overlayKps.length === POSE_TRACKED_IDS.length) {
          const media = (compareVideoRef.current ?? compareImageRef.current) as HTMLVideoElement | HTMLImageElement;
          const byId = new Map<number, {x: number; y: number; v: number}>();
          POSE_TRACKED_IDS.forEach((id, i) => {
            const p = overlayKps[i];
            if (p) byId.set(id, p);
          });

          ctx.save();
          const sideFromFacing = facingDirectionToLeg(compareFacingDirection);
          const trackedSide = compareKneeTrackingSide ?? sideFromFacing;
          const trackedIds = trackedSide === 'left' ? POSE_LEFT_SIDE_IDS : POSE_RIGHT_SIDE_IDS;
          const isTrackedConnection = (aId: number, bId: number) =>
            trackedIds.has(aId) && trackedIds.has(bId);
          const isArmConnection = (aId: number, bId: number) =>
            POSE_ARM_IDS.has(aId) || POSE_ARM_IDS.has(bId);
          const isHeadConnection = (aId: number, bId: number) =>
            aId === POSE_NOSE_ID || bId === POSE_NOSE_ID;

          for (const [aId, bId] of POSE_BODY_CONNECTIONS) {
            const a = byId.get(aId);
            const b = byId.get(bId);
            if (!a || !b || a.v < 0.25 || b.v < 0.25) continue;
            const head = isHeadConnection(aId, bId);
            const arm = isArmConnection(aId, bId);
            const tracked = !arm && !head && isTrackedConnection(aId, bId);
            ctx.strokeStyle = head ? 'rgba(0,210,255,0.15)' : arm ? 'rgba(0,210,255,0.15)' : tracked ? '#00d2ff' : 'rgba(0,210,255,0.25)';
            ctx.lineWidth = head ? 1 : arm ? 1 : tracked ? 3 : 1.5;
            const aPx = mapPoseNormToCanvasOverlayPx(a.x, a.y, media, canvas);
            const bPx = mapPoseNormToCanvasOverlayPx(b.x, b.y, media, canvas);
            ctx.beginPath();
            ctx.moveTo(aPx.x, aPx.y);
            ctx.lineTo(bPx.x, bPx.y);
            ctx.stroke();
          }

          POSE_TRACKED_IDS.forEach((id, i) => {
            const p = overlayKps[i];
            if (!p || p.v < 0.25) return;
            const arm = POSE_ARM_IDS.has(id);
            const nose = id === POSE_NOSE_ID;
            const tracked = !arm && !nose && trackedIds.has(id);
            ctx.fillStyle = nose ? 'rgba(0,210,255,0.4)' : arm ? 'rgba(0,210,255,0.15)' : tracked ? '#00d2ff' : 'rgba(0,210,255,0.25)';
            const o = mapPoseNormToCanvasOverlayPx(p.x, p.y, media, canvas);
            ctx.beginPath();
            ctx.arc(o.x, o.y, nose ? 3 : arm ? 2.5 : tracked ? 5 : 3.5, 0, Math.PI * 2);
            ctx.fill();
          });

          ctx.restore();
        }
      }

      for (const m of compareMeasurements) {
        if (m.kind === 'line' && m.points.length >= 2) {
          ctx.strokeStyle = accent;
          ctx.lineWidth = 3;
          const o0 = contentToOverlay(m.points[0].x, m.points[0].y, cZMap);
          const o1 = contentToOverlay(m.points[1].x, m.points[1].y, cZMap);
          ctx.beginPath();
          ctx.moveTo(o0.x, o0.y);
          ctx.lineTo(o1.x, o1.y);
          ctx.stroke();
          if (m.length !== null) {
            const midX = (o0.x + o1.x) / 2;
            const midY = (o0.y + o1.y) / 2;
            ctx.font = '20px Space Grotesk';
            ctx.fillStyle = accent;
            ctx.fillText(`${m.length.toFixed(0)} px`, midX + 8, midY - 8);
          }
        } else if (m.kind === 'angle' && m.points.length >= 2) {
          ctx.strokeStyle = accent;
          ctx.lineWidth = 3;
          const o0 = contentToOverlay(m.points[0].x, m.points[0].y, cZMap);
          const o1 = contentToOverlay(m.points[1].x, m.points[1].y, cZMap);
          ctx.beginPath();
          ctx.moveTo(o0.x, o0.y);
          ctx.lineTo(o1.x, o1.y);
          if (m.points.length === 3) {
            const o2 = contentToOverlay(m.points[2].x, m.points[2].y, cZMap);
            ctx.lineTo(o2.x, o2.y);
            ctx.font = '20px Space Grotesk';
            ctx.fillStyle = accent;
            ctx.fillText(`${m.angle?.toFixed(1)}°`, o1.x + 10, o1.y - 10);
          }
          ctx.stroke();
        }
      }
    };

    draw();

    const hasMeasurements = measurements.some(m => m.viewport === 'compare');
    if (compareVideoSrc && (poseEnabled || hasMeasurements)) {
      let rafId: number | null = null;
      let lastDrawnTime = -1;
      const loop = () => {
        const vt = compareVideoRef.current?.currentTime ?? -1;
        if (vt !== lastDrawnTime || hasMeasurements) {
          lastDrawnTime = vt;
          draw();
        }
        rafId = requestAnimationFrame(loop);
      };
      rafId = requestAnimationFrame(loop);
      return () => {
        if (rafId !== null) cancelAnimationFrame(rafId);
      };
    }
  }, [comparePoseKeypoints, compareVideoSrc, compareImageSrc, compareMediaLayoutVersion, poseEnabled, compareKneeTrackingSide, compareFacingDirection, measurements, compareAppliedZoom]);

  const [draggingPointIndex, setDraggingPointIndex] = useState<number | null>(null);
  const [compareCurrentTime, setCompareCurrentTime] = useState(0);
  const [compareDuration, setCompareDuration] = useState(0);
  const scrubRafRef = useRef<number | null>(null);
  const scrubTargetRef = useRef<number | null>(null);
  const compareScrubRafRef = useRef<number | null>(null);
  const compareScrubTargetRef = useRef<number | null>(null);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const fullscreenTargetRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const syncFullscreenState = () => {
      setIsFullscreen(document.fullscreenElement === fullscreenTargetRef.current);
    };
    document.addEventListener('fullscreenchange', syncFullscreenState);
    return () => document.removeEventListener('fullscreenchange', syncFullscreenState);
  }, []);

  const toggleFullscreen = async () => {
    const target = fullscreenTargetRef.current;
    if (!target) return;
    try {
      if (document.fullscreenElement === target) {
        await document.exitFullscreen();
      } else if (!document.fullscreenElement) {
        await target.requestFullscreen();
      } else {
        await document.exitFullscreen();
        await target.requestFullscreen();
      }
    } catch {
      /* Fullscreen may be blocked by browser policy or unsupported context */
    }
  };

  useEffect(() => {
    setCurrentTime(0);
    setDuration(0);

    const video = videoRef.current;
    if (!video) return;

    let lastSyncedDuration = 0;
    const syncDuration = () => {
      const d = getReliableVideoDuration(video);
      if (d > 0 && Math.abs(d - lastSyncedDuration) > 0.02) {
        lastSyncedDuration = d;
        setDuration(d);
      }
    };

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);

    const handleLoadedMetadata = () => syncDuration();
    const handleDurationChange = () => syncDuration();
    const handleProgress = () => syncDuration();
    const handleLoadedData = () => syncDuration();
    const handleCanPlay = () => syncDuration();
    const handleEnded = () => {
      const t = video.currentTime;
      if (Number.isFinite(t) && t > 0) {
        setDuration((prev) => Math.max(prev, t));
      }
      setCurrentTime(video.currentTime);
    };

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('durationchange', handleDurationChange);
    video.addEventListener('progress', handleProgress);
    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('canplay', handleCanPlay);
    video.addEventListener('ended', handleEnded);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('durationchange', handleDurationChange);
      video.removeEventListener('progress', handleProgress);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('canplay', handleCanPlay);
      video.removeEventListener('ended', handleEnded);
    };
  }, [videoSrc]);

  useEffect(() => {
    if (!poseEnabled) {
      setIsPoseAnalyzing(false);
      setGraphAnalysisRequested(false);
      analysisAbortRef.current = true;
      analysisGenRef.current += 1;
      setAnalysisProgress(null);
    }
  }, [poseEnabled]);

  const runFrameByFrameAnalysis = useCallback(async () => {
    const bgVideo = bgVideoRef.current;
    const mainVideo = videoRef.current;
    if (!bgVideo || !mainVideo) return;
    const detector = poseDetectorRef.current;
    if (!detector) return;

    const targetDuration = Math.max(getReliableVideoDuration(mainVideo), mainVideo.duration || 0, 0);
    if (targetDuration <= 0) return;

    let activeSide = facingDirectionToLeg(primaryFacingDirection);

    analysisAbortRef.current = true;
    const gen = ++analysisGenRef.current;
    const isStale = () => analysisGenRef.current !== gen;

    setKneeAngleSeries([]);
    setPosePointSeries([]);
    setBodyProportions(null);
    setCompareKneeAngleSeries([]);
    setComparePosePointSeries([]);
    setCompareBodyProportions(null);
    poseCacheRef.current = [];
    comparePoseCacheRef.current = [];
    setKneeTrackingSide(activeSide);
    setIsKneeGraphLocked(false);
    setIsPoseAnalyzing(true);
    setGraphDomainTime(targetDuration);
    setAnalysisProgress(0);
    analysisAbortRef.current = false;

    const ANALYSIS_FPS = 30;
    const frameStep = 1 / ANALYSIS_FPS;
    const PROGRESSIVE_BATCH = 15;

    const waitForVideoReady = (videoEl: HTMLVideoElement): Promise<boolean> =>
      new Promise((resolve) => {
        if (videoEl.readyState >= 2) { resolve(true); return; }
        let resolved = false;
        const finish = (ok: boolean) => {
          if (resolved) return;
          resolved = true;
          videoEl.removeEventListener('canplay', onReady);
          resolve(ok);
        };
        const onReady = () => finish(true);
        videoEl.addEventListener('canplay', onReady);
        setTimeout(() => finish(videoEl.readyState >= 2), 5000);
      });

    const seekAndWait = (videoEl: HTMLVideoElement, t: number, cap: number): Promise<void> =>
      new Promise((resolve) => {
        const clamped = Math.min(t, cap);
        if (Math.abs(videoEl.currentTime - clamped) < 0.005) {
          resolve();
          return;
        }
        let resolved = false;
        const finish = () => {
          if (resolved) return;
          resolved = true;
          videoEl.removeEventListener('seeked', finish);
          resolve();
        };
        videoEl.addEventListener('seeked', finish);
        videoEl.currentTime = clamped;
        setTimeout(finish, 2000);
      });

    const yieldToUI = () => new Promise<void>((r) => setTimeout(r, 0));
    const averageProportions = (samples: BodyProportions[]): ProportionProfile | null => {
      if (samples.length === 0) return null;
      const n = samples.length;
      const torsoLen = samples.reduce((s, p) => s + p.torsoLen, 0) / n;
      const femurLen = samples.reduce((s, p) => s + p.femurLen, 0) / n;
      const tibiaLen = samples.reduce((s, p) => s + p.tibiaLen, 0) / n;
      if (torsoLen < 1e-9 || femurLen < 1e-9 || tibiaLen < 1e-9) return null;
      const avg: BodyProportions = {
        torsoLen,
        femurLen,
        tibiaLen,
        femurToTorso: femurLen / torsoLen,
        tibiaToFemur: tibiaLen / femurLen,
        legToTorso: (femurLen + tibiaLen) / torsoLen,
      };
      return classifyProportions(avg);
    };

    const analyzeSingleVideo = async ({
      sourceVideo,
      visibleVideo,
      initialSide,
      setSeries,
      setPointSeries,
      cacheRef,
      setSide,
      setFacing,
      setProfile,
      progressOffset,
      progressScale,
    }: {
      sourceVideo: HTMLVideoElement;
      visibleVideo: HTMLVideoElement;
      initialSide: KneeSide;
      setSeries: (series: PoseAngleSample[]) => void;
      setPointSeries?: (series: PoseFrameSample[]) => void;
      cacheRef: MutableRefObject<{time: number; keypoints: {x: number; y: number; v: number}[]}[]>;
      setSide: (side: KneeSide) => void;
      setFacing?: (facing: FacingDirection) => void;
      setProfile: (profile: ProportionProfile | null) => void;
      progressOffset: number;
      progressScale: number;
    }) => {
      const localDuration = Math.max(getReliableVideoDuration(visibleVideo), visibleVideo.duration || 0, 0);
      if (localDuration <= 0) {
        cacheRef.current = [];
        setSeries([]);
        setPointSeries?.([]);
        setProfile(null);
        return;
      }
      const ready = await waitForVideoReady(sourceVideo);
      if (!ready || isStale()) return;

      let activeLocalSide = initialSide;
      let detectedFacing: FacingDirection | null = null;
      const sourceWidth = sourceVideo.videoWidth || visibleVideo.videoWidth;
      const sourceHeight = sourceVideo.videoHeight || visibleVideo.videoHeight;
      const ar = sourceHeight > 0 ? sourceWidth / sourceHeight : 1;
      const totalFrames = Math.ceil(localDuration * ANALYSIS_FPS);

      const angleSeries: PoseAngleSample[] = [];
      const pointSeries: PoseFrameSample[] = [];
      const cache: {time: number; keypoints: {x: number; y: number; v: number}[]}[] = [];
      const proportionSamples: BodyProportions[] = [];
      const analysisPoseCanvas = document.createElement('canvas');

      let t = 0;
      let frameIdx = 0;
      while (t <= localDuration && !isStale()) {
        await seekAndWait(sourceVideo, t, localDuration);
        if (isStale()) break;
        if (sourceVideo.readyState < 2) {
          await new Promise<void>((r) => {
            const onReady = () => { sourceVideo.removeEventListener('canplay', onReady); r(); };
            sourceVideo.addEventListener('canplay', onReady);
            setTimeout(() => { sourceVideo.removeEventListener('canplay', onReady); r(); }, 3000);
          });
        }

        const blazeposeInput = blazeposeVideoFrameCanvas(sourceVideo, analysisPoseCanvas);
        let normalized: {x: number; y: number; v: number}[] = [];
        if (blazeposeInput) {
          const result = await detector.estimatePoses(blazeposeInput, {flipHorizontal: false});
          if (isStale()) break;
          const keypoints = result?.[0]?.keypoints;
          const w = blazeposeInput.width;
          const h = blazeposeInput.height;
          if (keypoints && keypoints.length > 0 && w > 0 && h > 0) {
            normalized = keypointsFromBlazePoseOutput(keypoints, w, h);
          }
        }
        cache.push({time: t, keypoints: normalized});

        if (!detectedFacing && normalized.length === POSE_TRACKED_IDS.length) {
          const detected = detectFacingFromKeypoints(normalized);
          if (detected) {
            detectedFacing = detected;
            activeLocalSide = facingDirectionToLeg(detected);
            if (!isStale()) setSide(activeLocalSide);
            if (!isStale()) setFacing?.(detected);
          }
        }

        if (normalized.length === POSE_TRACKED_IDS.length) {
          const kneeAngle = getKneeAngleFromPose(normalized, activeLocalSide, MIN_POSE_VISIBILITY, ar);
          const anchors = getTrackedLegAnchors(normalized, activeLocalSide);
          const depth = getSquatDepthAngle(normalized, activeLocalSide, MIN_POSE_VISIBILITY, ar);
          const hipJointAngle = getHipAngle(normalized, activeLocalSide, MIN_POSE_VISIBILITY, ar);
          const torsoAngle = getBackAngle(normalized, activeLocalSide, MIN_POSE_VISIBILITY, ar);
          const frameProp = getBodyProportions(normalized, activeLocalSide, MIN_POSE_VISIBILITY, ar);
          if (frameProp) proportionSamples.push(frameProp);
          if (kneeAngle !== null && anchors) {
            const extension = Math.abs(anchors.ankleX - anchors.hipX);
            angleSeries.push({
              time: t,
              angle: kneeAngle,
              hipX: anchors.hipX,
              ankleX: anchors.ankleX,
              depthAngle: depth,
              hipAngle: hipJointAngle,
              backAngle: torsoAngle,
            });
            if (setPointSeries) {
              pointSeries.push({
                time: t,
                side: activeLocalSide,
                kneeAngle,
                hipX: anchors.hipX,
                ankleX: anchors.ankleX,
                extension,
                headY: getHeadNormYFromKeypoints(normalized),
                headToGroundY: getHeadToGroundNormGap(normalized, activeLocalSide),
                keypoints: normalized.map((p) => ({x: p.x, y: p.y, v: p.v})),
              });
            }
          }
        }

        frameIdx += 1;
        const pct = Math.min(100, Math.round((frameIdx / totalFrames) * 100));
        setAnalysisProgress(Math.min(100, Math.round(progressOffset + pct * progressScale)));

        if (frameIdx % PROGRESSIVE_BATCH === 0) {
          cacheRef.current = cache.slice();
          setSeries(angleSeries.slice());
          setPointSeries?.(pointSeries.slice());
          await yieldToUI();
        } else if (frameIdx % 2 === 0) {
          await yieldToUI();
        }
        t += frameStep;
      }

      if (isStale()) return;
      cacheRef.current = cache;
      setSeries(angleSeries);
      setPointSeries?.(pointSeries);
      setProfile(averageProportions(proportionSamples));
    };

    await analyzeSingleVideo({
      sourceVideo: bgVideo,
      visibleVideo: mainVideo,
      initialSide: activeSide,
      setSeries: setKneeAngleSeries,
      setPointSeries: setPosePointSeries,
      cacheRef: poseCacheRef,
      setSide: setKneeTrackingSide,
      setFacing: setPrimaryFacingDirection,
      setProfile: setBodyProportions,
      progressOffset: 0,
      progressScale: compareVideoSrc ? 0.5 : 1,
    });

    const compareBgVideo = compareBgVideoRef.current;
    const compareMainVideo = compareVideoRef.current;
    if (!isStale() && compareVideoSrc && compareBgVideo && compareMainVideo) {
      await analyzeSingleVideo({
        sourceVideo: compareBgVideo,
        visibleVideo: compareMainVideo,
        initialSide: compareKneeTrackingSide ?? facingDirectionToLeg(compareFacingDirection),
        setSeries: setCompareKneeAngleSeries,
        setPointSeries: setComparePosePointSeries,
        cacheRef: comparePoseCacheRef,
        setSide: setCompareKneeTrackingSide,
        setFacing: setCompareFacingDirection,
        setProfile: setCompareBodyProportions,
        progressOffset: 50,
        progressScale: 0.5,
      });
    } else if (!isStale()) {
      setCompareKneeAngleSeries([]);
      setComparePosePointSeries([]);
      comparePoseCacheRef.current = [];
      setCompareBodyProportions(null);
    }

    if (!isStale()) {
      setIsPoseAnalyzing(false);
      setIsKneeGraphLocked(true);
      setAnalysisProgress(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [primaryFacingDirection, compareFacingDirection, compareVideoSrc, compareKneeTrackingSide]);

  useEffect(() => {
    if (!graphAnalysisRequested) return;
    if (!poseEnabled || !videoSrc) return;
    if (poseStatus === 'error') {
      setIsPoseAnalyzing(false);
      setGraphAnalysisRequested(false);
      return;
    }
    if (poseStatus !== 'ready') return;
    if (!poseDetectorRef.current || !videoRef.current || !bgVideoRef.current) return;

    const mainVideo = videoRef.current;
    const targetDuration = Math.max(
      getReliableVideoDuration(mainVideo),
      mainVideo.duration || 0,
      0,
    );
    if (targetDuration <= 0) return;

    setGraphAnalysisRequested(false);
    void runFrameByFrameAnalysis();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [poseEnabled, poseStatus, videoSrc, graphAnalysisRequested, runFrameByFrameAnalysis]);

  useEffect(() => {
    setCompareCurrentTime(0);
    setCompareDuration(0);
    const video = compareVideoRef.current;
    if (!video) return;
    let lastSyncedDuration = 0;
    const syncDuration = () => {
      const d = getReliableVideoDuration(video);
      if (d > 0 && Math.abs(d - lastSyncedDuration) > 0.02) {
        lastSyncedDuration = d;
        setCompareDuration(d);
      }
    };
    const handleTimeUpdate = () => setCompareCurrentTime(video.currentTime);
    const handleLoadedMetadata = () => syncDuration();
    const handleDurationChange = () => syncDuration();
    const handleProgress = () => syncDuration();
    const handleLoadedData = () => syncDuration();
    const handleCanPlay = () => syncDuration();
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('durationchange', handleDurationChange);
    video.addEventListener('progress', handleProgress);
    video.addEventListener('loadeddata', handleLoadedData);
    video.addEventListener('canplay', handleCanPlay);
    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('durationchange', handleDurationChange);
      video.removeEventListener('progress', handleProgress);
      video.removeEventListener('loadeddata', handleLoadedData);
      video.removeEventListener('canplay', handleCanPlay);
    };
  }, [compareVideoSrc]);

  useEffect(() => {
    if (!poseEnabled) {
      setCurrentKneeAngle(null);
      setKneeAngleSeries([]);
      setPosePointSeries([]);
      setBodyProportions(null);
      setLiveBodyProportions(null);
      setCompareCurrentKneeAngle(null);
      setCompareCurrentLiveDepthAngle(null);
      setCompareCurrentLiveHipAngle(null);
      setCompareCurrentLiveBackAngle(null);
      setCompareKneeAngleSeries([]);
      setComparePosePointSeries([]);
      setCompareBodyProportions(null);
      setCompareLiveBodyProportions(null);
      poseCacheRef.current = [];
      comparePoseCacheRef.current = [];
      setKneeTrackingSide(null);
      setCompareKneeTrackingSide(null);
      setIsKneeGraphLocked(false);
      setIsPoseAnalyzing(false);
      setAnalysisProgress(null);
      analysisAbortRef.current = true;
      return;
    }
    if (!videoSrc) {
      setKneeAngleSeries([]);
      setPosePointSeries([]);
      setBodyProportions(null);
      poseCacheRef.current = [];
    }
    if (!compareVideoSrc) {
      setCompareKneeAngleSeries([]);
      setComparePosePointSeries([]);
      setCompareBodyProportions(null);
      comparePoseCacheRef.current = [];
    }
    if (!videoSrc && !imageSrc) {
      setKneeTrackingSide(null);
    }
    if (!compareVideoSrc && !compareImageSrc) {
      setCompareKneeTrackingSide(null);
    }
  }, [poseEnabled, videoSrc, imageSrc, compareVideoSrc, compareImageSrc]);

  useEffect(() => {
    setCurrentKneeAngle(null);
    setCurrentLiveDepthAngle(null);
    setCurrentLiveHipAngle(null);
    setCurrentLiveBackAngle(null);
    setKneeAngleSeries([]);
    setPosePointSeries([]);
    setBodyProportions(null);
    setLiveBodyProportions(null);
    setCompareCurrentKneeAngle(null);
    setCompareCurrentLiveDepthAngle(null);
    setCompareCurrentLiveHipAngle(null);
    setCompareCurrentLiveBackAngle(null);
    setCompareKneeAngleSeries([]);
    setComparePosePointSeries([]);
    setCompareBodyProportions(null);
    setCompareLiveBodyProportions(null);
    poseCacheRef.current = [];
    comparePoseCacheRef.current = [];
    setKneeTrackingSide(null);
    setCompareKneeTrackingSide(null);
    setGraphDomainTime(0);
    setAnalysisProgress(null);
    analysisAbortRef.current = true;
    setIsPoseAnalyzing(false);
  }, [videoSrc, imageSrc, compareVideoSrc, compareImageSrc]);

  useEffect(() => {
    if (!poseEnabled) {
      setCurrentKneeAngle(null);
      setCurrentLiveDepthAngle(null);
      setCurrentLiveHipAngle(null);
      setCurrentLiveBackAngle(null);
      setLiveBodyProportions(null);
      return;
    }
    const sideFromFacing = facingDirectionToLeg(primaryFacingDirection);
    const activeSide = kneeTrackingSide ?? sideFromFacing;
    if (!kneeTrackingSide) {
      setKneeTrackingSide(sideFromFacing);
    }
    const kps = poseCacheRef.current.length > 0
      ? (findNearestCachedPose(poseCacheRef.current, currentTime) ?? poseKeypoints)
      : poseKeypoints;
    const media = videoRef.current ?? imageRef.current;
    const w = media instanceof HTMLVideoElement ? media.videoWidth : media instanceof HTMLImageElement ? media.naturalWidth : 0;
    const h = media instanceof HTMLVideoElement ? media.videoHeight : media instanceof HTMLImageElement ? media.naturalHeight : 0;
    const liveAr = h > 0 ? w / h : 1;
    const kneeAngle = activeSide ? getKneeAngleFromPose(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null;
    setCurrentKneeAngle(kneeAngle);
    setCurrentLiveDepthAngle(activeSide ? getSquatDepthAngle(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
    setCurrentLiveHipAngle(activeSide ? getHipAngle(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
    setCurrentLiveBackAngle(activeSide ? getBackAngle(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
    setLiveBodyProportions(activeSide ? getBodyProportions(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
  }, [poseEnabled, poseKeypoints, currentTime, videoSrc, kneeTrackingSide, primaryFacingDirection]);

  useEffect(() => {
    if (!poseEnabled) {
      setCompareCurrentKneeAngle(null);
      setCompareCurrentLiveDepthAngle(null);
      setCompareCurrentLiveHipAngle(null);
      setCompareCurrentLiveBackAngle(null);
      setCompareLiveBodyProportions(null);
      return;
    }
    if (!compareVideoSrc && !compareImageSrc) {
      setCompareCurrentKneeAngle(null);
      setCompareCurrentLiveDepthAngle(null);
      setCompareCurrentLiveHipAngle(null);
      setCompareCurrentLiveBackAngle(null);
      setCompareLiveBodyProportions(null);
      return;
    }

    const sideFromFacing = facingDirectionToLeg(compareFacingDirection);
    const activeSide = compareKneeTrackingSide ?? sideFromFacing;
    if (!compareKneeTrackingSide) {
      setCompareKneeTrackingSide(sideFromFacing);
    }

    const kps = comparePoseCacheRef.current.length > 0
      ? (findNearestCachedPose(comparePoseCacheRef.current, compareCurrentTime) ?? comparePoseKeypoints)
      : comparePoseKeypoints;
    const media = compareVideoRef.current ?? compareImageRef.current;
    const w = media instanceof HTMLVideoElement ? media.videoWidth : media instanceof HTMLImageElement ? media.naturalWidth : 0;
    const h = media instanceof HTMLVideoElement ? media.videoHeight : media instanceof HTMLImageElement ? media.naturalHeight : 0;
    const liveAr = h > 0 ? w / h : 1;

    const kneeAngle = activeSide ? getKneeAngleFromPose(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null;
    setCompareCurrentKneeAngle(kneeAngle);
    setCompareCurrentLiveDepthAngle(activeSide ? getSquatDepthAngle(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
    setCompareCurrentLiveHipAngle(activeSide ? getHipAngle(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
    setCompareCurrentLiveBackAngle(activeSide ? getBackAngle(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
    setCompareLiveBodyProportions(activeSide ? getBodyProportions(kps, activeSide, MIN_POSE_VISIBILITY, liveAr) : null);
  }, [poseEnabled, comparePoseKeypoints, compareCurrentTime, compareVideoSrc, compareImageSrc, compareKneeTrackingSide, compareFacingDirection]);

  const handleSeek = (e: ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      const v = videoRef.current;
      const cap =
        duration > 0 ? duration : getReliableVideoDuration(v) || v.duration || time;
      const clamped = Number.isFinite(cap) && cap > 0 ? Math.min(Math.max(0, time), cap) : time;
      scrubTargetRef.current = clamped;
      if (scrubRafRef.current === null) {
        scrubRafRef.current = requestAnimationFrame(() => {
          scrubRafRef.current = null;
          if (!videoRef.current) return;
          const target = scrubTargetRef.current ?? 0;
          videoRef.current.currentTime = target;
          setCurrentTime(videoRef.current.currentTime);
        });
      }
    }
  };

  const handleCompareSeek = (e: ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (!compareVideoRef.current) return;
    const v = compareVideoRef.current;
    const cap =
      compareDuration > 0 ? compareDuration : getReliableVideoDuration(v) || v.duration || time;
    const clamped = Number.isFinite(cap) && cap > 0 ? Math.min(Math.max(0, time), cap) : time;
    compareScrubTargetRef.current = clamped;
    if (compareScrubRafRef.current === null) {
      compareScrubRafRef.current = requestAnimationFrame(() => {
        compareScrubRafRef.current = null;
        if (!compareVideoRef.current) return;
        const target = compareScrubTargetRef.current ?? 0;
        compareVideoRef.current.currentTime = target;
        setCompareCurrentTime(compareVideoRef.current.currentTime);
      });
    }
  };

  const handleDeleteMeasurements = () => {
    setMeasurements([]);
    setActiveMeasurementIdx(null);
    setIsDrawing(false);
    setActiveTool(null);
  };

  const runGraphAnalysis = () => {
    if (!videoSrc) return;
    analysisAbortRef.current = true;
    setIsKneeGraphLocked(false);
    if (!poseEnabled) setPoseEnabled(true);
    setGraphAnalysisRequested(true);
  };

  const isMediaLoaded = !!videoSrc || !!imageSrc;
  const hasCompareMedia = !!compareVideoSrc || !!compareImageSrc;
  const currentScale = appliedZoom?.s ?? 1;
  const compareScale = compareAppliedZoom?.s ?? 1;
  const activeScale = activeViewport === 'compare' && hasCompareMedia ? compareScale : currentScale;

  const pointerToOverlayPx = (clientX: number, clientY: number, canvas: HTMLCanvasElement) => {
    const rect = canvas.getBoundingClientRect();
    const rw = rect.width || 1;
    const rh = rect.height || 1;
    return {
      ox: ((clientX - rect.left) / rw) * canvas.width,
      oy: ((clientY - rect.top) / rh) * canvas.height,
    };
  };

  const pointerToContent = (clientX: number, clientY: number, canvas: HTMLCanvasElement) => {
    const {ox, oy} = pointerToOverlayPx(clientX, clientY, canvas);
    const zMap: AppliedZoomMap | null = appliedZoom
      ? {x: appliedZoom.x, y: appliedZoom.y, s: appliedZoom.s}
      : null;
    return overlayToContent(ox, oy, zMap);
  };

  const getPointAtOverlay = (ox: number, oy: number, viewport: ViewportTarget = 'primary'): { mIdx: number; pIdx: number } | null => {
    const zoom = viewport === 'primary' ? appliedZoom : compareAppliedZoom;
    const zMap: AppliedZoomMap | null = zoom
      ? {x: zoom.x, y: zoom.y, s: zoom.s}
      : null;
    for (let mIdx = 0; mIdx < measurements.length; mIdx++) {
      const m = measurements[mIdx];
      if (m.viewport !== viewport) continue;
      const pIdx = m.points.findIndex(p => {
        const o = contentToOverlay(p.x, p.y, zMap);
        return Math.hypot(o.x - ox, o.y - oy) < 15;
      });
      if (pIdx !== -1) return { mIdx, pIdx };
    }
    return null;
  };

  const getContentSize = (target: ViewportTarget = 'primary') => {
    const media = (
      target === 'primary'
        ? (videoRef.current ?? imageRef.current)
        : (compareVideoRef.current ?? compareImageRef.current)
    ) as HTMLElement | null;
    if (!media) return null;
    const w = media.clientWidth;
    const h = media.clientHeight;
    if (w <= 0 || h <= 0) return null;
    return {w, h};
  };

  const clampZoomViewport = (x: number, y: number, s: number, contentW: number, contentH: number) => {
    if (s <= ZOOM_MIN_SCALE) return {x: 0, y: 0, s: ZOOM_MIN_SCALE};
    const visibleW = contentW / s;
    const visibleH = contentH / s;
    const maxX = Math.max(0, contentW - visibleW);
    const maxY = Math.max(0, contentH - visibleH);
    return {x: clamp(x, 0, maxX), y: clamp(y, 0, maxY), s};
  };

  const zoomAt = (
    target: ViewportTarget,
    factor: number,
    anchorContentX: number,
    anchorContentY: number,
  ) => {
    const size = getContentSize(target);
    if (!size) return;
    const zoomState = target === 'primary' ? appliedZoom : compareAppliedZoom;
    const prevS = zoomState?.s ?? ZOOM_MIN_SCALE;
    const prevX = zoomState?.x ?? 0;
    const prevY = zoomState?.y ?? 0;
    const nextS = clamp(prevS * factor, ZOOM_MIN_SCALE, ZOOM_MAX_SCALE);
    if (Math.abs(nextS - ZOOM_MIN_SCALE) < 0.001) {
      if (target === 'primary') {
        setAppliedZoom(null);
      } else {
        setCompareAppliedZoom(null);
      }
      setActiveTool((prev) => (prev === 'pan' ? null : prev));
      return;
    }
    const nextX = anchorContentX - ((anchorContentX - prevX) * prevS) / nextS;
    const nextY = anchorContentY - ((anchorContentY - prevY) * prevS) / nextS;
    const clamped = clampZoomViewport(nextX, nextY, nextS, size.w, size.h);
    if (target === 'primary') {
      setAppliedZoom(clamped);
    } else {
      setCompareAppliedZoom(clamped);
    }
  };

  const zoomByButton = (factor: number) => {
    const target: ViewportTarget = activeViewport === 'compare' && hasCompareMedia ? 'compare' : 'primary';
    const size = getContentSize(target);
    if (!size) return;
    const centerOverlay = {
      ox: size.w / 2,
      oy: size.h / 2,
    };
    const anchor = overlayToContent(
      centerOverlay.ox,
      centerOverlay.oy,
      target === 'primary'
        ? (appliedZoom ? {x: appliedZoom.x, y: appliedZoom.y, s: appliedZoom.s} : null)
        : (compareAppliedZoom
            ? {x: compareAppliedZoom.x, y: compareAppliedZoom.y, s: compareAppliedZoom.s}
            : null),
    );
    zoomAt(target, factor, anchor.x, anchor.y);
  };

  const measurementNeedsNewSlot = (tool: 'angle' | 'line') => {
    const m = activeMeasurementIdx !== null ? measurements[activeMeasurementIdx] : null;
    if (m === null) return true;
    if (m.kind !== tool || m.viewport !== activeViewport) return true;
    if (tool === 'angle') return m.points.length >= 3;
    return m.points.length >= 2;
  };

  const toggleAngleTool = () => {
    setActiveTool((prev) => {
      if (prev === 'angle') {
        const m = activeMeasurementIdx !== null ? measurements[activeMeasurementIdx] : null;
        const currentComplete = m?.kind === 'angle' && (m.points.length ?? 0) >= 3;
        if (currentComplete) {
          const newIdx = measurements.length;
          setMeasurements(m_ => [...m_, { points: [], angle: null, length: null, kind: 'angle', viewport: activeViewport }]);
          setActiveMeasurementIdx(newIdx);
          setIsDrawing(true);
          return 'angle';
        }
        setIsDrawing(false);
        return null;
      }
      if (measurementNeedsNewSlot('angle')) {
        const newIdx = measurements.length;
        setMeasurements(m_ => [...m_, { points: [], angle: null, length: null, kind: 'angle', viewport: activeViewport }]);
        setActiveMeasurementIdx(newIdx);
      }
      setIsDrawing(true);
      return 'angle';
    });
  };

  const toggleLineTool = () => {
    setActiveTool((prev) => {
      if (prev === 'line') {
        const m = activeMeasurementIdx !== null ? measurements[activeMeasurementIdx] : null;
        const currentComplete = m?.kind === 'line' && (m.points.length ?? 0) >= 2;
        if (currentComplete) {
          const newIdx = measurements.length;
          setMeasurements(m_ => [...m_, { points: [], angle: null, length: null, kind: 'line', viewport: activeViewport }]);
          setActiveMeasurementIdx(newIdx);
          setIsDrawing(true);
          return 'line';
        }
        setIsDrawing(false);
        return null;
      }
      if (measurementNeedsNewSlot('line')) {
        const newIdx = measurements.length;
        setMeasurements(m_ => [...m_, { points: [], angle: null, length: null, kind: 'line', viewport: activeViewport }]);
        setActiveMeasurementIdx(newIdx);
      }
      setIsDrawing(true);
      return 'line';
    });
  };

  const togglePanTool = () => {
    setActiveTool((prev) => {
      const next = prev === 'pan' ? null : 'pan';
      if (next === 'pan') setIsDrawing(false);
      return next;
    });
  };

  const handleCanvasPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    setActiveViewport('primary');
    const canvas = canvasRef.current;
    if (!canvas) return;
    const {ox, oy} = pointerToOverlayPx(e.clientX, e.clientY, canvas);
    const {x, y} = pointerToContent(e.clientX, e.clientY, canvas);

    activePointersRef.current.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    const pointers = Array.from(activePointersRef.current.values()) as {clientX: number; clientY: number}[];
    if (e.pointerType === 'touch' && pointers.length >= 2) {
      const p0 = pointers[0]!;
      const p1 = pointers[1]!;
      const midClientX = (p0.clientX + p1.clientX) / 2;
      const midClientY = (p0.clientY + p1.clientY) / 2;
      const anchor = pointerToContent(midClientX, midClientY, canvas);
      const start = appliedZoom ?? {x: 0, y: 0, s: ZOOM_MIN_SCALE};
      pinchRef.current = {
        initialDistance: Math.hypot(p1.clientX - p0.clientX, p1.clientY - p0.clientY),
        initialScale: start.s,
        anchorX: anchor.x,
        anchorY: anchor.y,
        initialX: start.x,
        initialY: start.y,
      };
      panDragRef.current = null;
      setIsPanning(false);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    if (activeTool === 'pan' && currentScale > ZOOM_MIN_SCALE) {
      panDragRef.current = {
        pointerId: e.pointerId,
        startClientX: e.clientX,
        startClientY: e.clientY,
        startX: appliedZoom?.x ?? 0,
        startY: appliedZoom?.y ?? 0,
      };
      setIsPanning(true);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const hit = getPointAtOverlay(ox, oy, 'primary');
    if (hit) {
      e.currentTarget.setPointerCapture(e.pointerId);
      setActiveMeasurementIdx(hit.mIdx);
      setDraggingPointIndex(hit.pIdx);
      return;
    }

    if (!isDrawing || (activeTool !== 'angle' && activeTool !== 'line')) return;

    const currentMeasurement = activeMeasurementIdx !== null ? measurements[activeMeasurementIdx] : null;
    const tool = activeTool as 'angle' | 'line';
    const needNewMeasurement =
      !currentMeasurement ||
      currentMeasurement.viewport !== 'primary' ||
      currentMeasurement.kind !== tool ||
      (tool === 'angle' && currentMeasurement.points.length >= 3) ||
      (tool === 'line' && currentMeasurement.points.length >= 2);
    if (needNewMeasurement) {
      const newIdx = measurements.length;
      setMeasurements(m => [
        ...m,
        { points: [{ x, y }], angle: null, length: null, kind: tool, viewport: 'primary' },
      ]);
      setActiveMeasurementIdx(newIdx);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    e.currentTarget.setPointerCapture(e.pointerId);
    setPoints(prev => {
      const newPoints = [...prev, { x, y }];
      if (tool === 'angle' && newPoints.length === 3) {
        calculateAngle(newPoints);
      }
      if (tool === 'line' && newPoints.length === 2) {
        calculateLineLength(newPoints);
      }
      return newPoints;
    });
  };

  const handleCanvasPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (activePointersRef.current.has(e.pointerId)) {
      activePointersRef.current.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    }

    if (pinchRef.current) {
      const pointers = Array.from(activePointersRef.current.values()) as {clientX: number; clientY: number}[];
      if (pointers.length >= 2) {
        const p0 = pointers[0]!;
        const p1 = pointers[1]!;
        const distance = Math.hypot(p1.clientX - p0.clientX, p1.clientY - p0.clientY);
        if (distance > 1) {
          const size = getContentSize();
          if (!size) return;
          const pinch = pinchRef.current;
          const nextS = clamp(
            pinch.initialScale * (distance / Math.max(1, pinch.initialDistance)),
            ZOOM_MIN_SCALE,
            ZOOM_MAX_SCALE,
          );
          if (Math.abs(nextS - ZOOM_MIN_SCALE) < 0.001) {
            setAppliedZoom(null);
            setActiveTool((prev) => (prev === 'pan' ? null : prev));
            return;
          }
          const nextX = pinch.anchorX - ((pinch.anchorX - pinch.initialX) * pinch.initialScale) / nextS;
          const nextY = pinch.anchorY - ((pinch.anchorY - pinch.initialY) * pinch.initialScale) / nextS;
          setAppliedZoom(clampZoomViewport(nextX, nextY, nextS, size.w, size.h));
        }
      }
      return;
    }

    const {x, y} = pointerToContent(e.clientX, e.clientY, canvas);

    if (panDragRef.current && panDragRef.current.pointerId === e.pointerId && currentScale > ZOOM_MIN_SCALE) {
      const size = getContentSize();
      if (!size) return;
      const drag = panDragRef.current;
      const dx = e.clientX - drag.startClientX;
      const dy = e.clientY - drag.startClientY;
      setAppliedZoom(
        clampZoomViewport(drag.startX - dx / currentScale, drag.startY - dy / currentScale, currentScale, size.w, size.h),
      );
      return;
    }

    if (draggingPointIndex === null) return;

    setPoints(prev => {
      const newPoints = [...prev];
      newPoints[draggingPointIndex] = { x, y };
      const m = activeMeasurementIdx !== null ? measurements[activeMeasurementIdx] : null;
      if (m?.kind === 'angle' && newPoints.length === 3) {
        calculateAngle(newPoints);
      }
      if (m?.kind === 'line' && newPoints.length === 2) {
        calculateLineLength(newPoints);
      }
      return newPoints;
    });
  };

  const handleCanvasPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    activePointersRef.current.delete(e.pointerId);
    if (panDragRef.current?.pointerId === e.pointerId) {
      panDragRef.current = null;
      setIsPanning(false);
    }
    if (pinchRef.current && activePointersRef.current.size < 2) {
      pinchRef.current = null;
    }

    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore if not captured */
    }
    setDraggingPointIndex(null);
  };

  const handleCompareCanvasPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    setActiveViewport('compare');
    const canvas = compareCanvasRef.current;
    if (!canvas) return;

    const pointerToCompareContent = (clientX: number, clientY: number) => {
      const {ox, oy} = pointerToOverlayPx(clientX, clientY, canvas);
      const zMap: AppliedZoomMap | null = compareAppliedZoom
        ? {x: compareAppliedZoom.x, y: compareAppliedZoom.y, s: compareAppliedZoom.s}
        : null;
      return overlayToContent(ox, oy, zMap);
    };

    compareActivePointersRef.current.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    const pointers = Array.from(compareActivePointersRef.current.values()) as {clientX: number; clientY: number}[];
    if (e.pointerType === 'touch' && pointers.length >= 2) {
      const p0 = pointers[0]!;
      const p1 = pointers[1]!;
      const anchor = pointerToCompareContent((p0.clientX + p1.clientX) / 2, (p0.clientY + p1.clientY) / 2);
      const start = compareAppliedZoom ?? {x: 0, y: 0, s: ZOOM_MIN_SCALE};
      comparePinchRef.current = {
        initialDistance: Math.hypot(p1.clientX - p0.clientX, p1.clientY - p0.clientY),
        initialScale: start.s,
        anchorX: anchor.x,
        anchorY: anchor.y,
        initialX: start.x,
        initialY: start.y,
      };
      comparePanDragRef.current = null;
      setIsComparePanning(false);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    if (activeTool === 'pan' && compareScale > ZOOM_MIN_SCALE) {
      comparePanDragRef.current = {
        pointerId: e.pointerId,
        startClientX: e.clientX,
        startClientY: e.clientY,
        startX: compareAppliedZoom?.x ?? 0,
        startY: compareAppliedZoom?.y ?? 0,
      };
      setIsComparePanning(true);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    const {ox, oy} = pointerToOverlayPx(e.clientX, e.clientY, canvas);
    const {x, y} = pointerToCompareContent(e.clientX, e.clientY);

    const hit = getPointAtOverlay(ox, oy, 'compare');
    if (hit) {
      e.currentTarget.setPointerCapture(e.pointerId);
      setActiveMeasurementIdx(hit.mIdx);
      setDraggingPointIndex(hit.pIdx);
      return;
    }

    if (!isDrawing || (activeTool !== 'angle' && activeTool !== 'line')) return;

    const currentMeasurement = activeMeasurementIdx !== null ? measurements[activeMeasurementIdx] : null;
    const tool = activeTool as 'angle' | 'line';
    const needNewMeasurement =
      !currentMeasurement ||
      currentMeasurement.viewport !== 'compare' ||
      currentMeasurement.kind !== tool ||
      (tool === 'angle' && currentMeasurement.points.length >= 3) ||
      (tool === 'line' && currentMeasurement.points.length >= 2);
    if (needNewMeasurement) {
      const newIdx = measurements.length;
      setMeasurements(m => [
        ...m,
        { points: [{ x, y }], angle: null, length: null, kind: tool, viewport: 'compare' },
      ]);
      setActiveMeasurementIdx(newIdx);
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }

    e.currentTarget.setPointerCapture(e.pointerId);
    setPoints(prev => {
      const newPoints = [...prev, { x, y }];
      if (tool === 'angle' && newPoints.length === 3) {
        calculateAngle(newPoints);
      }
      if (tool === 'line' && newPoints.length === 2) {
        calculateLineLength(newPoints);
      }
      return newPoints;
    });
  };

  const handleCompareCanvasPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = compareCanvasRef.current;
    if (!canvas) return;
    if (compareActivePointersRef.current.has(e.pointerId)) {
      compareActivePointersRef.current.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    }

    if (comparePinchRef.current) {
      const pointers = Array.from(compareActivePointersRef.current.values()) as {clientX: number; clientY: number}[];
      if (pointers.length >= 2) {
        const p0 = pointers[0]!;
        const p1 = pointers[1]!;
        const distance = Math.hypot(p1.clientX - p0.clientX, p1.clientY - p0.clientY);
        if (distance > 1) {
          const size = getContentSize('compare');
          if (!size) return;
          const pinch = comparePinchRef.current;
          const nextS = clamp(
            pinch.initialScale * (distance / Math.max(1, pinch.initialDistance)),
            ZOOM_MIN_SCALE,
            ZOOM_MAX_SCALE,
          );
          if (Math.abs(nextS - ZOOM_MIN_SCALE) < 0.001) {
            setCompareAppliedZoom(null);
            return;
          }
          const nextX = pinch.anchorX - ((pinch.anchorX - pinch.initialX) * pinch.initialScale) / nextS;
          const nextY = pinch.anchorY - ((pinch.anchorY - pinch.initialY) * pinch.initialScale) / nextS;
          setCompareAppliedZoom(clampZoomViewport(nextX, nextY, nextS, size.w, size.h));
        }
      }
      return;
    }

    if (comparePanDragRef.current && comparePanDragRef.current.pointerId === e.pointerId && compareScale > ZOOM_MIN_SCALE) {
      const size = getContentSize('compare');
      if (!size) return;
      const drag = comparePanDragRef.current;
      const dx = e.clientX - drag.startClientX;
      const dy = e.clientY - drag.startClientY;
      setCompareAppliedZoom(
        clampZoomViewport(drag.startX - dx / compareScale, drag.startY - dy / compareScale, compareScale, size.w, size.h),
      );
      return;
    }

    if (draggingPointIndex === null) return;

    const {ox: _ox, oy: _oy} = pointerToOverlayPx(e.clientX, e.clientY, canvas);
    const compareZMap: AppliedZoomMap | null = compareAppliedZoom
      ? {x: compareAppliedZoom.x, y: compareAppliedZoom.y, s: compareAppliedZoom.s}
      : null;
    const contentPt = overlayToContent(_ox, _oy, compareZMap);

    setPoints(prev => {
      const newPoints = [...prev];
      newPoints[draggingPointIndex] = { x: contentPt.x, y: contentPt.y };
      const m = activeMeasurementIdx !== null ? measurements[activeMeasurementIdx] : null;
      if (m?.kind === 'angle' && newPoints.length === 3) {
        calculateAngle(newPoints);
      }
      if (m?.kind === 'line' && newPoints.length === 2) {
        calculateLineLength(newPoints);
      }
      return newPoints;
    });
  };

  const handleCompareCanvasPointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    compareActivePointersRef.current.delete(e.pointerId);
    if (comparePanDragRef.current?.pointerId === e.pointerId) {
      comparePanDragRef.current = null;
      setIsComparePanning(false);
    }
    if (comparePinchRef.current && compareActivePointersRef.current.size < 2) {
      comparePinchRef.current = null;
    }
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore if not captured */
    }
    setDraggingPointIndex(null);
  };

  const calculateAngle = (pts: { x: number; y: number }[]) => {
    const [p1, p2, p3] = pts;
    setAngle(calculateJointAngle(p1, p2, p3));
  };

  const calculateLineLength = (pts: { x: number; y: number }[]) => {
    if (pts.length < 2) return;
    const [p0, p1] = pts;
    setLength(Math.hypot(p1.x - p0.x, p1.y - p0.y));
  };

  const handleAnalysisModeSwitch = (mode: AnalysisMode) => {
    setAnalysisMode(mode);
    if (mode === 'squat') {
      setAngleLowerBound(40);
      setAngleUpperBound(120);
    } else {
      setAngleLowerBound(100);
      setAngleUpperBound(180);
    }
  };

  const renderKneeAnglePanel = () => {
    if (!isMediaLoaded) return null;

    const isSquat = analysisMode === 'squat';
    const series = kneeAngleSeries;
    const toKneeFlexion = (jointAngle: number | null) => jointAngle === null ? null : 180 - jointAngle;
    const toHipFlexionProxy = (jointAngle: number | null) => jointAngle === null ? null : 180 - jointAngle;
    const formatAngle = (value: number | null) => value !== null ? `${value.toFixed(1)}°` : '—';

    const computePanelStats = (
      panelSeries: PoseAngleSample[],
      panelTime: number,
      panelCurrentKneeAngle: number | null,
      panelLiveDepth: number | null,
      panelLiveHip: number | null,
      panelLiveBack: number | null,
      panelFacing: FacingDirection,
      poseFrames: PoseFrameSample[],
      trackSide: KneeSide,
      videoAspect: number,
    ) => {
      const panelAllMaxima = isSquat ? [] : detectKneeAngleMaxima(panelSeries, {
        lowerAngle: angleLowerBound,
        upperAngle: angleUpperBound,
        facing: panelFacing,
      });
      const panelMaxima = panelAllMaxima.filter((p) => p.direction === extensionFilter);
      const panelOtherMaxima = panelAllMaxima.filter((p) => p.direction !== extensionFilter);
      const panelAvgPeakAngle =
        panelMaxima.length > 0
          ? panelMaxima.reduce((sum, peak) => sum + peak.angle, 0) / panelMaxima.length
          : null;
      const panelMaxPeakAngle =
        panelMaxima.length > 0
          ? Math.max(...panelMaxima.map((p) => p.angle))
          : null;

      const panelValleys = isSquat ? detectKneeAngleMinima(panelSeries, {
        lowerAngle: angleLowerBound,
        upperAngle: angleUpperBound,
      }) : [];
      const panelBelowParallelCount = panelValleys.filter((v) => v.belowParallel).length;
      const panelAvgRepKneeAngle =
        panelValleys.length > 0
          ? panelValleys.reduce((sum, v) => sum + v.angle, 0) / panelValleys.length
          : null;
      const panelAvgDepthAngle =
        panelValleys.length > 0
          ? panelValleys.reduce((sum, v) => sum + v.depthAngle, 0) / panelValleys.length
          : null;
      const panelAvgRepHipAngle = (() => {
        const withHip = panelValleys.filter((v) => v.hipAngle !== null);
        if (withHip.length === 0) return null;
        return withHip.reduce((sum, v) => sum + v.hipAngle!, 0) / withHip.length;
      })();
      const panelAvgRepBackAngle = (() => {
        const withBack = panelValleys.filter((v) => v.backAngle !== null);
        if (withBack.length === 0) return null;
        return withBack.reduce((sum, v) => sum + v.backAngle!, 0) / withBack.length;
      })();
      const panelNearest = (() => {
        if (!isSquat || panelSeries.length === 0) return null;
        return panelSeries.reduce((best, s) =>
          Math.abs(s.time - panelTime) < Math.abs(best.time - panelTime) ? s : best
        );
      })();
      const panelCurrentDepthAngle = panelLiveDepth ?? panelNearest?.depthAngle ?? null;
      const panelCurrentHipAngle = panelLiveHip ?? panelNearest?.hipAngle ?? null;
      const panelCurrentBackAngle = panelLiveBack ?? panelNearest?.backAngle ?? null;

      const contactToeMarkTimes: number[] = [];
      if (!isSquat && panelMaxima.length > 0) {
        for (const peak of panelMaxima) {
          contactToeMarkTimes.push(
            findContactTimeAfterExtensionPeak(panelSeries, peak.time) ?? peak.time,
          );
        }
      }

      let avgContactToeAngle: number | null = null;
      if (!isSquat && poseFrames.length > 0 && panelMaxima.length > 0) {
        const toeAngles: number[] = [];
        for (const peak of panelMaxima) {
          const contactT =
            findContactTimeAfterExtensionPeak(panelSeries, peak.time) ?? peak.time;
          const frame = nearestPoseFrameByTime(poseFrames, contactT);
          if (!frame) continue;
          const deg = getFootInclinationVsHorizontalDeg(frame.keypoints, trackSide, videoAspect);
          if (deg !== null) toeAngles.push(deg);
        }
        avgContactToeAngle =
          toeAngles.length > 0 ? toeAngles.reduce((a, b) => a + b, 0) / toeAngles.length : null;
      }

      let currentHeadVerticalPct: number | null = null;
      let avgHeadMovementPerStridePct: number | null = null;
      if (!isSquat && poseFrames.length > 0) {
        const gapList = poseFrames
          .map((f) => f.headToGroundY)
          .filter((g): g is number => g !== null && g !== undefined && g > 1e-9 && Number.isFinite(g));
        const medianGap = medianFinite(gapList);
        const nf = nearestPoseFrameByTime(poseFrames, panelTime);
        let curGap = nf?.headToGroundY ?? null;
        if (curGap === null && nf?.keypoints.length === POSE_TRACKED_IDS.length) {
          curGap = getHeadToGroundNormGap(nf.keypoints, trackSide);
        }
        currentHeadVerticalPct =
          medianGap !== null && medianGap > 1e-9 && curGap !== null && curGap > 1e-9
            ? (curGap / medianGap) * 100
            : null;
        avgHeadMovementPerStridePct = computeAvgHeadMovementPerStrideVsGroundPct(
          poseFrames,
          contactToeMarkTimes,
          panelMaxima.map((p) => p.time),
        );
      }

      return {
        allMaxima: panelAllMaxima,
        maxima: panelMaxima,
        otherMaxima: panelOtherMaxima,
        avgPeakAngle: panelAvgPeakAngle,
        maxPeakAngle: panelMaxPeakAngle,
        valleys: panelValleys,
        belowParallelCount: panelBelowParallelCount,
        avgRepKneeAngle: panelAvgRepKneeAngle,
        avgDepthAngle: panelAvgDepthAngle,
        avgRepHipAngle: panelAvgRepHipAngle,
        avgRepBackAngle: panelAvgRepBackAngle,
        currentDepthAngle: panelCurrentDepthAngle,
        currentHipAngle: panelCurrentHipAngle,
        currentBackAngle: panelCurrentBackAngle,
        currentKneeDisplay: toKneeFlexion(panelCurrentKneeAngle),
        avgKneeDisplay: toKneeFlexion(panelAvgRepKneeAngle),
        currentHipDisplay: toHipFlexionProxy(panelCurrentHipAngle),
        avgHipDisplay: toHipFlexionProxy(panelAvgRepHipAngle),
        avgContactToeAngle,
        contactToeMarkTimes,
        currentHeadVerticalPct,
        avgHeadMovementPerStridePct,
      };
    };

    const primaryVideoAspect =
      videoRef.current && videoRef.current.videoHeight > 0
        ? videoRef.current.videoWidth / videoRef.current.videoHeight
        : imageRef.current && imageRef.current.naturalHeight > 0
          ? imageRef.current.naturalWidth / imageRef.current.naturalHeight
          : 1;
    const compareVideoAspect =
      compareVideoRef.current && compareVideoRef.current.videoHeight > 0
        ? compareVideoRef.current.videoWidth / compareVideoRef.current.videoHeight
        : compareImageRef.current && compareImageRef.current.naturalHeight > 0
          ? compareImageRef.current.naturalWidth / compareImageRef.current.naturalHeight
          : 1;

    const primaryTrackSide = kneeTrackingSide ?? facingDirectionToLeg(primaryFacingDirection);
    const compareTrackSide =
      compareKneeTrackingSide ?? facingDirectionToLeg(compareFacingDirection);

    const primaryStats = computePanelStats(
      series,
      currentTime,
      currentKneeAngle,
      currentLiveDepthAngle,
      currentLiveHipAngle,
      currentLiveBackAngle,
      primaryFacingDirection,
      posePointSeries,
      primaryTrackSide,
      primaryVideoAspect,
    );
    const compareStats = hasCompareMedia
      ? computePanelStats(
        compareKneeAngleSeries,
        compareCurrentTime,
        compareCurrentKneeAngle,
        compareCurrentLiveDepthAngle,
        compareCurrentLiveHipAngle,
        compareCurrentLiveBackAngle,
        compareFacingDirection,
        comparePosePointSeries,
        compareTrackSide,
        compareVideoAspect,
      )
      : null;

    const allMaxima = primaryStats.allMaxima;
    const maxima = primaryStats.maxima;
    const otherMaxima = primaryStats.otherMaxima;
    const avgPeakAngle = primaryStats.avgPeakAngle;
    const maxPeakAngle = primaryStats.maxPeakAngle;
    const valleys = primaryStats.valleys;
    const belowParallelCount = primaryStats.belowParallelCount;
    const avgRepKneeAngle = primaryStats.avgRepKneeAngle;
    const avgDepthAngle = primaryStats.avgDepthAngle;
    const avgRepHipAngle = primaryStats.avgRepHipAngle;
    const avgRepBackAngle = primaryStats.avgRepBackAngle;
    const currentDepthAngle = primaryStats.currentDepthAngle;
    const currentHipAngle = primaryStats.currentHipAngle;
    const currentBackAngle = primaryStats.currentBackAngle;
    const currentKneeDisplay = primaryStats.currentKneeDisplay;
    const avgKneeDisplay = primaryStats.avgKneeDisplay;
    const currentHipDisplay = primaryStats.currentHipDisplay;
    const avgHipDisplay = primaryStats.avgHipDisplay;
    const compareMaxima = compareStats?.maxima ?? [];
    const compareOtherMaxima = compareStats?.otherMaxima ?? [];
    const compareValleys = compareStats?.valleys ?? [];
    const avgContactToeAngle = primaryStats.avgContactToeAngle;
    const compareAvgContactToeAngle = compareStats?.avgContactToeAngle ?? null;
    const contactToeMarkTimes = primaryStats.contactToeMarkTimes;
    const compareContactToeMarkTimes = compareStats?.contactToeMarkTimes ?? [];
    const currentHeadVerticalPct = primaryStats.currentHeadVerticalPct;
    const avgHeadMovementPerStridePct = primaryStats.avgHeadMovementPerStridePct;
    const compareCurrentHeadVerticalPct = compareStats?.currentHeadVerticalPct ?? null;
    const compareAvgHeadMovementPerStridePct = compareStats?.avgHeadMovementPerStridePct ?? null;

    const primaryToeAtPlayhead = !isSquat
      ? (() => {
          if (posePointSeries.length > 0) {
            const nf = nearestPoseFrameByTime(posePointSeries, currentTime);
            if (nf && nf.keypoints.length === POSE_TRACKED_IDS.length) {
              const d = getFootInclinationVsHorizontalDeg(nf.keypoints, primaryTrackSide, primaryVideoAspect);
              if (d !== null) return d;
            }
          }
          if (poseKeypoints.length === POSE_TRACKED_IDS.length) {
            return getFootInclinationVsHorizontalDeg(poseKeypoints, primaryTrackSide, primaryVideoAspect);
          }
          return null;
        })()
      : null;
    const compareToeAtPlayhead =
      hasCompareMedia && !isSquat
        ? (() => {
            if (comparePosePointSeries.length > 0) {
              const nf = nearestPoseFrameByTime(comparePosePointSeries, compareCurrentTime);
              if (nf && nf.keypoints.length === POSE_TRACKED_IDS.length) {
                const d = getFootInclinationVsHorizontalDeg(nf.keypoints, compareTrackSide, compareVideoAspect);
                if (d !== null) return d;
              }
            }
            if (comparePoseKeypoints.length === POSE_TRACKED_IDS.length) {
              return getFootInclinationVsHorizontalDeg(comparePoseKeypoints, compareTrackSide, compareVideoAspect);
            }
            return null;
          })()
        : null;

    const compareSeries = compareKneeAngleSeries;
    const seriesEndTime = series.length > 0 ? series[series.length - 1]!.time : 0;
    const compareSeriesEndTime = compareSeries.length > 0 ? compareSeries[compareSeries.length - 1]!.time : 0;
    const sliderTime = Math.max(duration, currentTime, 0.0001);
    const compareSliderTime = Math.max(compareDuration, compareCurrentTime, 0.0001);
    const maxTime = graphDomainTime > 0 ? graphDomainTime : Math.max(sliderTime, seriesEndTime, 0.0001);
    const compareMaxTime = graphDomainTime > 0 ? graphDomainTime : Math.max(compareSliderTime, compareSeriesEndTime, 0.0001);
    const graphHeight = 140;

    const strideMinAngle = 0;
    const strideMaxAngle = 180;

    const depthAngles = isSquat
      ? [...series, ...(hasCompareMedia ? compareSeries : [])]
        .map((s) => s.depthAngle)
        .filter((d): d is number => d !== null)
      : [];
    const depthMinRaw = depthAngles.length > 0 ? Math.min(...depthAngles) : -30;
    const depthMaxRaw = depthAngles.length > 0 ? Math.max(...depthAngles) : 80;
    const sqMinAngle = Math.min(depthMinRaw - 5, -10);
    const sqMaxAngle = Math.max(depthMaxRaw + 5, 10);

    const gMin = isSquat ? sqMinAngle : strideMinAngle;
    const gMax = isSquat ? sqMaxAngle : strideMaxAngle;

    const lowerY = clamp(
      graphHeight - ((angleLowerBound - gMin) / (gMax - gMin)) * graphHeight,
      0,
      graphHeight,
    );
    const upperY = clamp(
      graphHeight - ((angleUpperBound - gMin) / (gMax - gMin)) * graphHeight,
      0,
      graphHeight,
    );
    const parallelY = isSquat
      ? clamp(graphHeight - ((0 - gMin) / (gMax - gMin)) * graphHeight, 0, graphHeight)
      : 0;
    const updateBoundFromPointer = (clientY: number, svg: SVGSVGElement | null, bound: 'lower' | 'upper') => {
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      if (rect.height <= 0) return;
      const yPx = clamp(clientY - rect.top, 0, rect.height);
      const yNorm = yPx / rect.height;
      const nextAngle = gMax - yNorm * (gMax - gMin);
      if (bound === 'lower') {
        setAngleLowerBound(clamp(nextAngle, gMin, angleUpperBound - 1));
      } else {
        setAngleUpperBound(clamp(nextAngle, angleLowerBound + 1, gMax));
      }
    };

    const seekFromGraphPointer = (clientX: number, svg: SVGSVGElement | null) => {
      if (!svg || !videoRef.current) return;
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0) return;
      const xNorm = clamp((clientX - rect.left) / rect.width, 0, 1);
      const time = xNorm * maxTime;
      scrubTargetRef.current = time;
      if (scrubRafRef.current === null) {
        scrubRafRef.current = requestAnimationFrame(() => {
          scrubRafRef.current = null;
          if (!videoRef.current) return;
          const target = scrubTargetRef.current ?? 0;
          videoRef.current.currentTime = target;
          setCurrentTime(videoRef.current.currentTime);
        });
      }
      setCurrentTime(time);
    };

    const handleGraphPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
      setDraggingPlayhead(true);
      wasPlayingBeforeScrubRef.current = isPlaying;
      if (isPlaying) setIsPlaying(false);
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      seekFromGraphPointer(e.clientX, graphSvgRef.current);
    };

    const handleGraphPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
      if (!draggingPlayhead) return;
      seekFromGraphPointer(e.clientX, graphSvgRef.current);
    };

    const handleGraphPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
      if (!draggingPlayhead) return;
      setDraggingPlayhead(false);
      if (wasPlayingBeforeScrubRef.current) setIsPlaying(true);
      try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch { /* */ }
    };

    const seekFromCompareGraphPointer = (clientX: number, svg: SVGSVGElement | null) => {
      if (!svg || !compareVideoRef.current) return;
      const rect = svg.getBoundingClientRect();
      if (rect.width <= 0) return;
      const xNorm = clamp((clientX - rect.left) / rect.width, 0, 1);
      const time = xNorm * compareMaxTime;
      compareScrubTargetRef.current = time;
      if (compareScrubRafRef.current === null) {
        compareScrubRafRef.current = requestAnimationFrame(() => {
          compareScrubRafRef.current = null;
          if (!compareVideoRef.current) return;
          const target = compareScrubTargetRef.current ?? 0;
          compareVideoRef.current.currentTime = target;
          setCompareCurrentTime(compareVideoRef.current.currentTime);
        });
      }
      setCompareCurrentTime(time);
    };

    const handleCompareGraphPointerDown = (e: ReactPointerEvent<SVGSVGElement>) => {
      setDraggingComparePlayhead(true);
      wasPlayingBeforeCompareScrubRef.current = isPlaying;
      if (isPlaying) setIsPlaying(false);
      (e.currentTarget as Element).setPointerCapture(e.pointerId);
      seekFromCompareGraphPointer(e.clientX, compareGraphSvgRef.current);
    };

    const handleCompareGraphPointerMove = (e: ReactPointerEvent<SVGSVGElement>) => {
      if (!draggingComparePlayhead) return;
      seekFromCompareGraphPointer(e.clientX, compareGraphSvgRef.current);
    };

    const handleCompareGraphPointerUp = (e: ReactPointerEvent<SVGSVGElement>) => {
      if (!draggingComparePlayhead) return;
      setDraggingComparePlayhead(false);
      if (wasPlayingBeforeCompareScrubRef.current) setIsPlaying(true);
      try { (e.currentTarget as Element).releasePointerCapture(e.pointerId); } catch { /* */ }
    };

    const playheadX = maxTime > 0 ? (currentTime / maxTime) * 100 : 0;
    const comparePlayheadX = compareMaxTime > 0 ? (compareCurrentTime / compareMaxTime) * 100 : 0;

    const makeLinePath = (panelSeries: PoseAngleSample[], panelMaxTime: number) =>
      panelSeries
        .map((sample, idx) => {
          const x = (sample.time / panelMaxTime) * 100;
          const val = isSquat ? (sample.depthAngle ?? 0) : sample.angle;
          const y = graphHeight - ((val - gMin) / (gMax - gMin)) * graphHeight;
          return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(3)} ${y.toFixed(3)}`;
        })
        .join(' ');

    const linePath = makeLinePath(series, maxTime);
    const compareLinePath = makeLinePath(compareSeries, compareMaxTime);

    return (
      <section className={`rounded-xl border p-3 sm:p-4 ${isFullscreen ? 'border-transparent bg-black/50 backdrop-blur-md' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]'}`}>
        <div className="mb-2 flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold uppercase tracking-wide text-[var(--color-text-light)]">
            {isSquat ? 'Squat Depth — Below Parallel' : 'Knee Angle (Hip-Knee-Ankle)'}
          </h3>
          <div className="flex items-center gap-2">
            {!isSquat && (
              <button
                type="button"
                onClick={() => setExtensionFilter((prev) => (prev === 'forward' ? 'behind' : 'forward'))}
                className="rounded-md border border-[var(--color-accent)]/20 px-2 py-1 text-xs hover:bg-[var(--color-panel-hover)] text-[var(--color-accent)]"
                title="Toggle forward / behind extension filter"
              >
                {extensionFilter === 'forward' ? 'Forward' : 'Behind'}
              </button>
            )}
            <div className="flex rounded-md border border-[var(--color-accent)]/20 overflow-hidden">
              <button
                type="button"
                onClick={() => handleAnalysisModeSwitch('stride')}
                className={`px-2 py-1 text-xs transition-colors ${analysisMode === 'stride' ? 'bg-[var(--color-accent)] text-[var(--color-bg-dark)] font-semibold' : 'text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)]'}`}
                title="Stride analysis — track extension peaks"
              >
                Stride
              </button>
              <button
                type="button"
                onClick={() => handleAnalysisModeSwitch('squat')}
                className={`px-2 py-1 text-xs transition-colors ${analysisMode === 'squat' ? 'bg-[var(--color-accent)] text-[var(--color-bg-dark)] font-semibold' : 'text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)]'}`}
                title="Squat analysis — track depth valleys and 90° threshold"
              >
                Squat
              </button>
            </div>
            {videoSrc ? (
              <button
                type="button"
                onClick={runGraphAnalysis}
                className="rounded-md border border-[var(--color-accent)]/20 p-1.5 hover:bg-[var(--color-panel-hover)] text-[var(--color-accent)]"
                title="Analyze graph"
                aria-label="Analyze graph"
              >
                <LineChart className="h-4 w-4" />
              </button>
            ) : null}
          </div>
        </div>
        {videoSrc ? (
          <div className={`grid gap-2 ${hasCompareMedia ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
            <div className={`relative rounded-lg border ${isFullscreen ? 'border-transparent bg-black/40' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/70'}`}>
            {series.length >= 2 ? (
              <>
              <svg
                ref={graphSvgRef}
                viewBox={`0 0 100 ${graphHeight}`}
                preserveAspectRatio="none"
                className="h-36 w-full"
                style={{ cursor: draggingPlayhead ? 'grabbing' : 'pointer', touchAction: 'none' }}
                role="img"
                aria-label={isSquat ? 'Squat depth over time' : 'Knee angle over time'}
                onPointerDown={handleGraphPointerDown}
                onPointerMove={handleGraphPointerMove}
                onPointerUp={handleGraphPointerUp}
                onPointerCancel={handleGraphPointerUp}
              >
                <path
                  d={linePath}
                  fill="none"
                  stroke="var(--color-accent)"
                  strokeWidth={0.8}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  vectorEffect="non-scaling-stroke"
                />
                {isSquat ? (
                  <line x1={0} y1={parallelY} x2={100} y2={parallelY} stroke="#ffffff" strokeWidth={0.55} strokeDasharray="2 1.2" opacity={0.5} vectorEffect="non-scaling-stroke" />
                ) : (
                  <>
                    <rect
                      x={0}
                      y={upperY}
                      width={100}
                      height={Math.max(0, lowerY - upperY)}
                      fill="var(--color-accent)"
                      opacity={0.06}
                    />
                    <line x1={0} y1={lowerY} x2={100} y2={lowerY} stroke="#ff4d4f" strokeWidth={0.55} strokeDasharray="2 1.2" opacity={0.95} vectorEffect="non-scaling-stroke" />
                    <line
                      x1={0} y1={lowerY} x2={100} y2={lowerY}
                      stroke="transparent" strokeWidth={8} className="cursor-ns-resize"
                      onPointerDown={(e) => { e.stopPropagation(); setDraggingBound('lower'); e.currentTarget.setPointerCapture(e.pointerId); updateBoundFromPointer(e.clientY, e.currentTarget.ownerSVGElement, 'lower'); }}
                      onPointerMove={(e) => { if (draggingBound === 'lower') updateBoundFromPointer(e.clientY, e.currentTarget.ownerSVGElement, 'lower'); }}
                      onPointerUp={(e) => { setDraggingBound(null); try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* */ } }}
                      onPointerCancel={(e) => { setDraggingBound(null); try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* */ } }}
                    />
                    <line x1={0} y1={upperY} x2={100} y2={upperY} stroke="#52c41a" strokeWidth={0.55} strokeDasharray="2 1.2" opacity={0.95} vectorEffect="non-scaling-stroke" />
                    <line
                      x1={0} y1={upperY} x2={100} y2={upperY}
                      stroke="transparent" strokeWidth={8} className="cursor-ns-resize"
                      onPointerDown={(e) => { e.stopPropagation(); setDraggingBound('upper'); e.currentTarget.setPointerCapture(e.pointerId); updateBoundFromPointer(e.clientY, e.currentTarget.ownerSVGElement, 'upper'); }}
                      onPointerMove={(e) => { if (draggingBound === 'upper') updateBoundFromPointer(e.clientY, e.currentTarget.ownerSVGElement, 'upper'); }}
                      onPointerUp={(e) => { setDraggingBound(null); try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* */ } }}
                      onPointerCancel={(e) => { setDraggingBound(null); try { e.currentTarget.releasePointerCapture(e.pointerId); } catch { /* */ } }}
                    />
                  </>
                )}
                {!isSquat && otherMaxima.map((peak) => {
                  const x = (peak.time / maxTime) * 100;
                  return (
                    <line
                      key={`other-${peak.time.toFixed(4)}`}
                      x1={x}
                      y1={0}
                      x2={x}
                      y2={graphHeight}
                      stroke="#555"
                      strokeWidth={0.25}
                      strokeDasharray="1.25 1.15"
                      opacity={0.4}
                      vectorEffect="non-scaling-stroke"
                    />
                  );
                })}
                {!isSquat && maxima.map((peak) => {
                  const x = (peak.time / maxTime) * 100;
                  return (
                    <line
                      key={`peak-${peak.time.toFixed(4)}`}
                      x1={x}
                      y1={0}
                      x2={x}
                      y2={graphHeight}
                      stroke="#00d2ff"
                      strokeWidth={0.35}
                      strokeDasharray="1.25 1.15"
                      opacity={0.95}
                      vectorEffect="non-scaling-stroke"
                    />
                  );
                })}
                {!isSquat &&
                  contactToeMarkTimes.map((t, i) => {
                    const x = maxTime > 0 ? (t / maxTime) * 100 : 0;
                    return (
                      <line
                        key={`contact-toe-${i}-${t.toFixed(4)}`}
                        x1={x}
                        y1={0}
                        x2={x}
                        y2={graphHeight}
                        stroke="#facc15"
                        strokeWidth={0.35}
                        strokeDasharray="1.25 1.15"
                        opacity={0.92}
                        vectorEffect="non-scaling-stroke"
                        style={{ pointerEvents: 'none' }}
                      />
                    );
                  })}
                {isSquat && valleys.map((valley) => {
                  const x = (valley.time / maxTime) * 100;
                  return (
                    <line
                      key={`valley-${valley.time.toFixed(4)}`}
                      x1={x}
                      y1={0}
                      x2={x}
                      y2={graphHeight}
                      stroke={valley.belowParallel ? '#52c41a' : '#ff4d4f'}
                      strokeWidth={0.4}
                      strokeDasharray="1.25 1.15"
                      opacity={0.95}
                      vectorEffect="non-scaling-stroke"
                    />
                  );
                })}
                {maxTime > 0 && series.length >= 2 && (
                  <line
                    x1={playheadX}
                    y1={0}
                    x2={playheadX}
                    y2={graphHeight}
                    stroke="white"
                    strokeWidth={1.5}
                    opacity={draggingPlayhead ? 1 : 0.7}
                    vectorEffect="non-scaling-stroke"
                    style={{ pointerEvents: 'none' }}
                  />
                )}
              </svg>
              {isSquat ? (
                <span
                  className="pointer-events-none absolute left-1 text-[10px] font-medium leading-none"
                  style={{bottom: `${((0 - gMin) / (gMax - gMin)) * 100}%`, color: '#ffffff', opacity: 0.5, transform: 'translateY(-2px)'}}
                >
                  0° parallel
                </span>
              ) : (
                <>
                  <span
                    className="pointer-events-none absolute left-1 text-[10px] font-medium leading-none"
                    style={{bottom: `${((angleUpperBound - gMin) / (gMax - gMin)) * 100}%`, color: '#52c41a', transform: 'translateY(-2px)'}}
                  >
                    {angleUpperBound.toFixed(0)}°
                  </span>
                  <span
                    className="pointer-events-none absolute left-1 text-[10px] font-medium leading-none"
                    style={{bottom: `${((angleLowerBound - gMin) / (gMax - gMin)) * 100}%`, color: '#ff4d4f', transform: 'translateY(-2px)'}}
                  >
                    {angleLowerBound.toFixed(0)}°
                  </span>
                </>
              )}
              {isSquat && series.length >= 2 && (
                <div className="pointer-events-none absolute bottom-1 right-1 flex items-center gap-2 rounded bg-black/20 px-1.5 py-0.5 text-[10px] text-[var(--color-text-light)]">
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm" style={{background: '#52c41a'}} /> below</span>
                  <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm" style={{background: '#ff4d4f'}} /> above</span>
                </div>
              )}
              {!isSquat && series.length >= 2 && (
                <div className="pointer-events-none absolute bottom-1 right-1 flex flex-wrap items-center justify-end gap-x-3 gap-y-1 rounded bg-black/25 px-1.5 py-0.5 text-[10px] text-[var(--color-text-light)]">
                  <span className="flex items-center gap-1">
                    <svg width={28} height={10} viewBox="0 0 28 10" className="shrink-0" aria-hidden>
                      <line x1={0} y1={5} x2={28} y2={5} stroke="#00d2ff" strokeWidth={2} strokeDasharray="4 3" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
                    </svg>
                    extension peak
                  </span>
                  <span className="flex items-center gap-1">
                    <svg width={28} height={10} viewBox="0 0 28 10" className="shrink-0" aria-hidden>
                      <line x1={0} y1={5} x2={28} y2={5} stroke="#facc15" strokeWidth={2} strokeDasharray="4 3" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
                    </svg>
                    toe @ contact
                  </span>
                </div>
              )}
              </>
            ) : (
              <p className="py-8 text-center text-sm text-[var(--color-text-light)]">
                {isSquat
                  ? 'Play or scrub the video to collect squat-depth samples.'
                  : 'Play or scrub the video to collect knee-angle samples.'}
              </p>
            )}
            </div>
            {hasCompareMedia && (
              <div className={`relative rounded-lg border ${isFullscreen ? 'border-transparent bg-black/40' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/70'}`}>
                {compareSeries.length >= 2 ? (
                  <>
                    <svg
                      ref={compareGraphSvgRef}
                      viewBox={`0 0 100 ${graphHeight}`}
                      preserveAspectRatio="none"
                      className="h-36 w-full"
                      style={{ cursor: draggingComparePlayhead ? 'grabbing' : 'pointer', touchAction: 'none' }}
                      role="img"
                      aria-label={isSquat ? 'Compare squat depth over time' : 'Compare knee angle over time'}
                      onPointerDown={handleCompareGraphPointerDown}
                      onPointerMove={handleCompareGraphPointerMove}
                      onPointerUp={handleCompareGraphPointerUp}
                      onPointerCancel={handleCompareGraphPointerUp}
                    >
                      <path
                        d={compareLinePath}
                        fill="none"
                        stroke="var(--color-accent)"
                        strokeWidth={0.8}
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        vectorEffect="non-scaling-stroke"
                      />
                      {isSquat ? (
                        <line x1={0} y1={parallelY} x2={100} y2={parallelY} stroke="#ffffff" strokeWidth={0.55} strokeDasharray="2 1.2" opacity={0.5} vectorEffect="non-scaling-stroke" />
                      ) : (
                        <>
                          <rect
                            x={0}
                            y={upperY}
                            width={100}
                            height={Math.max(0, lowerY - upperY)}
                            fill="var(--color-accent)"
                            opacity={0.06}
                          />
                          <line x1={0} y1={lowerY} x2={100} y2={lowerY} stroke="#ff4d4f" strokeWidth={0.55} strokeDasharray="2 1.2" opacity={0.95} vectorEffect="non-scaling-stroke" />
                          <line x1={0} y1={upperY} x2={100} y2={upperY} stroke="#52c41a" strokeWidth={0.55} strokeDasharray="2 1.2" opacity={0.95} vectorEffect="non-scaling-stroke" />
                        </>
                      )}
                      {!isSquat && compareOtherMaxima.map((peak) => {
                        const x = (peak.time / compareMaxTime) * 100;
                        return (
                          <line
                            key={`compare-other-${peak.time.toFixed(4)}`}
                            x1={x}
                            y1={0}
                            x2={x}
                            y2={graphHeight}
                            stroke="#555"
                            strokeWidth={0.25}
                            strokeDasharray="1.25 1.15"
                            opacity={0.4}
                            vectorEffect="non-scaling-stroke"
                          />
                        );
                      })}
                      {!isSquat && compareMaxima.map((peak) => {
                        const x = (peak.time / compareMaxTime) * 100;
                        return (
                          <line
                            key={`compare-peak-${peak.time.toFixed(4)}`}
                            x1={x}
                            y1={0}
                            x2={x}
                            y2={graphHeight}
                            stroke="#00d2ff"
                            strokeWidth={0.35}
                            strokeDasharray="1.25 1.15"
                            opacity={0.95}
                            vectorEffect="non-scaling-stroke"
                          />
                        );
                      })}
                      {!isSquat &&
                        compareContactToeMarkTimes.map((t, i) => {
                          const x = compareMaxTime > 0 ? (t / compareMaxTime) * 100 : 0;
                          return (
                            <line
                              key={`compare-contact-toe-${i}-${t.toFixed(4)}`}
                              x1={x}
                              y1={0}
                              x2={x}
                              y2={graphHeight}
                              stroke="#facc15"
                              strokeWidth={0.35}
                              strokeDasharray="1.25 1.15"
                              opacity={0.92}
                              vectorEffect="non-scaling-stroke"
                              style={{ pointerEvents: 'none' }}
                            />
                          );
                        })}
                      {isSquat && compareValleys.map((valley) => {
                        const x = (valley.time / compareMaxTime) * 100;
                        return (
                          <line
                            key={`compare-valley-${valley.time.toFixed(4)}`}
                            x1={x}
                            y1={0}
                            x2={x}
                            y2={graphHeight}
                            stroke={valley.belowParallel ? '#52c41a' : '#ff4d4f'}
                            strokeWidth={0.4}
                            strokeDasharray="1.25 1.15"
                            opacity={0.95}
                            vectorEffect="non-scaling-stroke"
                          />
                        );
                      })}
                      {compareMaxTime > 0 && compareSeries.length >= 2 && (
                        <line
                          x1={comparePlayheadX}
                          y1={0}
                          x2={comparePlayheadX}
                          y2={graphHeight}
                          stroke="white"
                          strokeWidth={1.5}
                          opacity={draggingComparePlayhead ? 1 : 0.7}
                          vectorEffect="non-scaling-stroke"
                          style={{ pointerEvents: 'none' }}
                        />
                      )}
                    </svg>
                    {isSquat ? (
                      <span
                        className="pointer-events-none absolute left-1 text-[10px] font-medium leading-none"
                        style={{bottom: `${((0 - gMin) / (gMax - gMin)) * 100}%`, color: '#ffffff', opacity: 0.5, transform: 'translateY(-2px)'}}
                      >
                        0° parallel
                      </span>
                    ) : (
                      <>
                        <span
                          className="pointer-events-none absolute left-1 text-[10px] font-medium leading-none"
                          style={{bottom: `${((angleUpperBound - gMin) / (gMax - gMin)) * 100}%`, color: '#52c41a', transform: 'translateY(-2px)'}}
                        >
                          {angleUpperBound.toFixed(0)}°
                        </span>
                        <span
                          className="pointer-events-none absolute left-1 text-[10px] font-medium leading-none"
                          style={{bottom: `${((angleLowerBound - gMin) / (gMax - gMin)) * 100}%`, color: '#ff4d4f', transform: 'translateY(-2px)'}}
                        >
                          {angleLowerBound.toFixed(0)}°
                        </span>
                      </>
                    )}
                    {isSquat && compareSeries.length >= 2 && (
                      <div className="pointer-events-none absolute bottom-1 right-1 flex items-center gap-2 rounded bg-black/20 px-1.5 py-0.5 text-[10px] text-[var(--color-text-light)]">
                        <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm" style={{background: '#52c41a'}} /> below</span>
                        <span className="flex items-center gap-1"><span className="inline-block h-2 w-3 rounded-sm" style={{background: '#ff4d4f'}} /> above</span>
                      </div>
                    )}
                    {!isSquat && compareSeries.length >= 2 && (
                      <div className="pointer-events-none absolute bottom-1 right-1 flex flex-wrap items-center justify-end gap-x-3 gap-y-1 rounded bg-black/25 px-1.5 py-0.5 text-[10px] text-[var(--color-text-light)]">
                        <span className="flex items-center gap-1">
                          <svg width={28} height={10} viewBox="0 0 28 10" className="shrink-0" aria-hidden>
                            <line x1={0} y1={5} x2={28} y2={5} stroke="#00d2ff" strokeWidth={2} strokeDasharray="4 3" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
                          </svg>
                          extension peak
                        </span>
                        <span className="flex items-center gap-1">
                          <svg width={28} height={10} viewBox="0 0 28 10" className="shrink-0" aria-hidden>
                            <line x1={0} y1={5} x2={28} y2={5} stroke="#facc15" strokeWidth={2} strokeDasharray="4 3" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
                          </svg>
                          toe @ contact
                        </span>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="py-8 text-center text-sm text-[var(--color-text-light)]">
                    {isSquat
                      ? 'Play or scrub the compare video to collect squat-depth samples.'
                      : 'Play or scrub the compare video to collect knee-angle samples.'}
                  </p>
                )}
              </div>
            )}
          </div>
        ) : (
          <p className="text-sm text-[var(--color-text-light)]">
            Angle graph is available for video playback; enable Pose to start tracking.
          </p>
        )}
        {analysisProgress !== null && (
          <div className="mt-2">
            <div className="flex items-center justify-between text-xs text-[var(--color-text-light)] mb-1">
              <span>Analyzing frames... {analysisProgress}%</span>
            </div>
            <div className="h-1.5 w-full rounded-full bg-[var(--color-bg-dark)] overflow-hidden border border-[var(--color-accent)]/10">
              <div
                className="h-full rounded-full bg-[var(--color-accent)] transition-[width] duration-150"
                style={{width: `${analysisProgress}%`}}
              />
            </div>
          </div>
        )}
        {isSquat ? (
          <div className={`mt-2 grid gap-2 ${hasCompareMedia ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
            <div className={`rounded-md border p-2.5 ${isFullscreen ? 'border-transparent bg-black/50 backdrop-blur-md' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/40'}`}>
              <div className="grid grid-cols-5 gap-x-3 text-xs text-[var(--color-text-light)]">
                <p className="font-medium text-[var(--color-accent)]">Depth (coach): {formatAngle(currentDepthAngle)}</p>
                <p className="font-medium text-[var(--color-accent)]">Knee flex: {formatAngle(currentKneeDisplay)}</p>
                <p className="font-medium text-[var(--color-accent)]">Hip flex*: {formatAngle(currentHipDisplay)}</p>
                <p className="font-medium text-[var(--color-accent)]">Trunk lean: {formatAngle(currentBackAngle)}</p>
                <p className="font-medium text-[var(--color-accent)]">Reps: {valleys.length}</p>
              </div>
              <div className="mt-1 grid grid-cols-5 gap-x-3 text-xs text-[var(--color-text-light)]">
                <p>Avg: {formatAngle(avgDepthAngle)}</p>
                <p>Avg: {formatAngle(avgKneeDisplay)}</p>
                <p>Avg: {formatAngle(avgHipDisplay)}</p>
                <p>Avg: {formatAngle(avgRepBackAngle)}</p>
                <p>Below parallel: {belowParallelCount}/{valleys.length}</p>
              </div>
            </div>
            {hasCompareMedia && (
              <div className={`rounded-md border p-2.5 ${isFullscreen ? 'border-transparent bg-black/50 backdrop-blur-md' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/40'}`}>
                <p className="mb-1 text-[10px] uppercase tracking-wide opacity-60 text-[var(--color-text-light)]">Compare</p>
                <div className="grid grid-cols-5 gap-x-3 text-xs text-[var(--color-text-light)]">
                  <p className="font-medium text-[var(--color-accent)]">Depth (coach): {formatAngle(compareStats?.currentDepthAngle ?? null)}</p>
                  <p className="font-medium text-[var(--color-accent)]">Knee flex: {formatAngle(compareStats?.currentKneeDisplay ?? null)}</p>
                  <p className="font-medium text-[var(--color-accent)]">Hip flex*: {formatAngle(compareStats?.currentHipDisplay ?? null)}</p>
                  <p className="font-medium text-[var(--color-accent)]">Trunk lean: {formatAngle(compareStats?.currentBackAngle ?? null)}</p>
                  <p className="font-medium text-[var(--color-accent)]">Reps: {compareStats?.valleys.length ?? 0}</p>
                </div>
                <div className="mt-1 grid grid-cols-5 gap-x-3 text-xs text-[var(--color-text-light)]">
                  <p>Avg: {formatAngle(compareStats?.avgDepthAngle ?? null)}</p>
                  <p>Avg: {formatAngle(compareStats?.avgKneeDisplay ?? null)}</p>
                  <p>Avg: {formatAngle(compareStats?.avgHipDisplay ?? null)}</p>
                  <p>Avg: {formatAngle(compareStats?.avgRepBackAngle ?? null)}</p>
                  <p>Below parallel: {(compareStats?.belowParallelCount ?? 0)}/{compareStats?.valleys.length ?? 0}</p>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className={`mt-2 grid gap-2 ${hasCompareMedia ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
            <div className={`rounded-md border p-2.5 ${isFullscreen ? 'border-transparent bg-black/50 backdrop-blur-md' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/40'}`}>
              <div className="grid grid-cols-3 gap-x-2 gap-y-1 text-xs text-[var(--color-text-light)]">
                <p className="font-medium text-[var(--color-accent)]" title="Hip–knee–ankle at playhead">
                  Knee angle: {currentKneeAngle !== null ? `${currentKneeAngle.toFixed(1)}°` : '—'}
                </p>
                <p className="font-medium text-[var(--color-accent)]" title="Heel–toe vs horizon at playhead">
                  Toe angle: {formatAngle(primaryToeAtPlayhead)}
                </p>
                <p
                  className="font-medium text-[var(--color-accent)]"
                  title="Nose→foot (lower heel/toe) gap vs median gap in this run (100 ≈ typical)"
                >
                  Head vs ground: {currentHeadVerticalPct !== null ? `${currentHeadVerticalPct.toFixed(1)}%` : '—'}
                </p>
              </div>
              <div className="mt-1 grid grid-cols-3 gap-x-2 gap-y-1 text-xs text-[var(--color-text-light)]">
                <p title="Mean knee angle at extension peaks (filtered)">
                  Avg at extension: {avgPeakAngle !== null ? `${avgPeakAngle.toFixed(1)}°` : '—'}
                </p>
                <p title="Mean toe angle at ground contact (after each peak)">
                  Avg at contact: {formatAngle(avgContactToeAngle)}
                </p>
                <p title="Mean head bob per stride as % of mean nose→foot distance in that stride">
                  Avg movement / stride: {avgHeadMovementPerStridePct !== null ? `${avgHeadMovementPerStridePct.toFixed(1)}%` : '—'}
                </p>
              </div>
            </div>
            {hasCompareMedia && (
              <div className={`rounded-md border p-2.5 ${isFullscreen ? 'border-transparent bg-black/50 backdrop-blur-md' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/40'}`}>
                <p className="mb-1 text-[10px] uppercase tracking-wide opacity-60 text-[var(--color-text-light)]">Compare</p>
                <div className="grid grid-cols-3 gap-x-2 gap-y-1 text-xs text-[var(--color-text-light)]">
                  <p className="font-medium text-[var(--color-accent)]" title="Hip–knee–ankle at playhead">
                    Knee angle: {compareCurrentKneeAngle !== null ? `${compareCurrentKneeAngle.toFixed(1)}°` : '—'}
                  </p>
                  <p className="font-medium text-[var(--color-accent)]" title="Heel–toe vs horizon at playhead">
                    Toe angle: {formatAngle(compareToeAtPlayhead)}
                  </p>
                  <p
                    className="font-medium text-[var(--color-accent)]"
                    title="Nose→foot (lower heel/toe) gap vs median gap in this run (100 ≈ typical)"
                  >
                    Head vs ground: {compareCurrentHeadVerticalPct !== null ? `${compareCurrentHeadVerticalPct.toFixed(1)}%` : '—'}
                  </p>
                </div>
                <div className="mt-1 grid grid-cols-3 gap-x-2 gap-y-1 text-xs text-[var(--color-text-light)]">
                  <p title="Mean knee angle at extension peaks (filtered)">
                    Avg at extension: {compareStats?.avgPeakAngle !== null && compareStats?.avgPeakAngle !== undefined ? `${compareStats.avgPeakAngle.toFixed(1)}°` : '—'}
                  </p>
                  <p title="Mean toe angle at ground contact (after each peak)">
                    Avg at contact: {formatAngle(compareAvgContactToeAngle)}
                  </p>
                  <p title="Mean head bob per stride as % of mean nose→foot distance in that stride">
                    Avg movement / stride: {compareAvgHeadMovementPerStridePct !== null ? `${compareAvgHeadMovementPerStridePct.toFixed(1)}%` : '—'}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}
        {showAngleGuide && (
          <div className={`mt-2 rounded-md border p-2.5 ${isFullscreen ? 'border-transparent bg-black/50 backdrop-blur-md' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/40'}`}>
            <div className="mb-1.5 flex items-center justify-between">
              <p className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
                Angle guide
              </p>
            </div>
            {isSquat ? (
            <div className="grid gap-2 text-[10px] text-[var(--color-text-light)] md:grid-cols-2 xl:grid-cols-4">
              <div className="rounded border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50 p-2">
                <p className="font-semibold text-[var(--color-accent)]">Knee flex</p>
                <p className="opacity-75">How bent the knee is at the bottom. Higher means deeper bend.</p>
                <p className="mt-1 font-mono opacity-60">hip •───∠knee───• ankle</p>
              </div>
              <div className="rounded border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50 p-2">
                <p className="font-semibold text-[var(--color-accent)]">Depth (coach)</p>
                <p className="opacity-75">Hip-to-knee segment vs horizontal. 0° is parallel; below 0° is below parallel.</p>
                <p className="mt-1 font-mono opacity-60">thigh angle vs ─────────</p>
              </div>
              <div className="rounded border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50 p-2">
                <p className="font-semibold text-[var(--color-accent)]">Hip flex*</p>
                <p className="opacity-75">Trunk-thigh proxy for hip bend. Higher means more closed hip position.</p>
                <p className="mt-1 font-mono opacity-60">shoulder •─∠hip─• knee</p>
              </div>
              <div className="rounded border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50 p-2">
                <p className="font-semibold text-[var(--color-accent)]">Trunk lean</p>
                <p className="opacity-75">Torso inclination from vertical. Higher means more forward torso lean.</p>
                <p className="mt-1 font-mono opacity-60">vertical │ vs shoulder↘hip</p>
              </div>
            </div>
          ) : (
            <div className="grid gap-2 text-[10px] text-[var(--color-text-light)] md:grid-cols-3">
              <div className="rounded border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50 p-2">
                <p className="font-semibold text-[var(--color-accent)]">Knee angle @ extension</p>
                <p className="opacity-75">Current at playhead; average is mean knee angle at extension peaks (direction filter).</p>
                <p className="mt-1 font-mono opacity-60">hip •───∠knee───• ankle</p>
              </div>
              <div className="rounded border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50 p-2">
                <p className="font-semibold text-[var(--color-accent)]">Toe @ ground contact</p>
                <p className="opacity-75">
                  Acute heel–toe vs horizon at contact (first knee flex after each peak). Average is over contacts.
                </p>
                <p className="mt-1 font-mono opacity-60">heel ●──→ toe vs ────</p>
              </div>
              <div className="rounded border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50 p-2">
                <p className="font-semibold text-[var(--color-accent)]">Head vs ground / stride</p>
                <p className="opacity-75">
                  Ground = lowest heel/toe Y (same as toe calc). Head vs ground compares nose→foot gap to the median gap
                  in this clip. Movement per stride is head bob divided by mean nose→foot distance within that stride.
                </p>
                <p className="mt-1 font-mono opacity-60">nose→foot gap · bob / gap</p>
              </div>
            </div>
          )}
          </div>
        )}
        {isSquat && isPoseAnalyzing && (
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[var(--color-text-light)]">
            <span className="text-xs">analyzing...</span>
          </div>
        )}
        {isSquat && showBiomechanics && (() => {
          const primaryProfile = bodyProportions;
          const primaryLive = liveBodyProportions;
          const primaryDisplay = primaryProfile ?? (primaryLive ? classifyProportions(primaryLive) : null);
          const primaryProps = primaryDisplay?.proportions;
          const primaryShowLive = !primaryProfile && primaryLive;

          const compareProfile = compareBodyProportions;
          const compareLive = compareLiveBodyProportions;
          const compareDisplay = compareProfile ?? (compareLive ? classifyProportions(compareLive) : null);
          const compareProps = compareDisplay?.proportions;
          const compareShowLive = !compareProfile && compareLive;

          if (!primaryDisplay || !primaryProps) return null;

          type SegCat = ProportionProfile['femurCategory'];
          const catColor = (cat: SegCat) => {
            const deviationMap: Record<SegCat, number> = {
              short: 2,
              'slightly-short': 1,
              average: 0,
              'slightly-long': 1,
              long: 2,
            };
            const deviation = deviationMap[cat];
            if (deviation === 0) return '#52c41a';
            if (deviation === 1) return '#faad14';
            return '#ff4d4f';
          };

          const bioPanelClass = `rounded-lg border p-3 ${
            isFullscreen
              ? 'border-transparent bg-black/50 backdrop-blur-md'
              : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/50'
          }`;
          const bioInnerCellClass = `rounded-md border border-[var(--color-accent)]/10 px-2 py-1.5 ${
            isFullscreen ? 'bg-[var(--color-bg-dark)]/50' : 'bg-[var(--color-bg-dark)]'
          }`;

          const renderProportionCard = (profile: ProportionProfile, props: BodyProportions, showLive: BodyProportions | false | null) => {
            const leanDeg = profile.estimatedLeanDeg;
            const parallelKneeFlex = profile.estimatedParallelKneeFlexDeg;
            const parallelHipFlex = profile.estimatedParallelHipFlexDeg;
            return (
              <div className={bioPanelClass}>
                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
                  Biomechanical @ parallel
                </p>
                <div className="mb-2.5 grid grid-cols-1 gap-2 text-center text-[11px] text-[var(--color-text-light)] sm:grid-cols-3">
                  <div className={bioInnerCellClass}>
                    <p className="text-[9px] uppercase tracking-wider opacity-50">Knee flex @ parallel</p>
                    <p className="text-sm font-semibold text-[var(--color-accent)]">{parallelKneeFlex.toFixed(1)}°</p>
                  </div>
                  <div className={bioInnerCellClass}>
                    <p className="text-[9px] uppercase tracking-wider opacity-50">Hip flex* @ parallel</p>
                    <p className="text-sm font-semibold text-[var(--color-accent)]">{parallelHipFlex.toFixed(1)}°</p>
                  </div>
                  <div className={bioInnerCellClass}>
                    <p className="text-[9px] uppercase tracking-wider opacity-50">Trunk lean @ parallel</p>
                    <p className="text-sm font-semibold text-[var(--color-accent)]">{leanDeg.toFixed(1)}°</p>
                  </div>
                </div>

                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
                  Proportions {showLive ? <span className="font-normal opacity-55">(snapshot)</span> : <span className="font-normal opacity-55">(average)</span>}
                </p>
                <div className="mb-2.5 grid grid-cols-2 gap-2 text-center text-[11px] text-[var(--color-text-light)]">
                  <div className={bioInnerCellClass}>
                    <p className="text-[9px] uppercase tracking-wider opacity-50">Femur / Torso</p>
                    <p className="text-sm font-semibold text-[var(--color-accent)]">{props.femurToTorso.toFixed(2)}</p>
                    <p className="text-[9px] font-medium" style={{color: catColor(profile.femurCategory)}}>
                      {profile.femurCategory.replace('-', ' ')} femurs
                    </p>
                  </div>
                  <div className={bioInnerCellClass}>
                    <p className="text-[9px] uppercase tracking-wider opacity-50">Tibia / Femur</p>
                    <p className="text-sm font-semibold text-[var(--color-accent)]">{props.tibiaToFemur.toFixed(2)}</p>
                    <p className="text-[9px] font-medium" style={{color: catColor(profile.tibiaCategory)}}>
                      {profile.tibiaCategory.replace('-', ' ')} tibias
                    </p>
                  </div>
                </div>

                <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
                  Advice
                </p>
                <div className="space-y-1">
                  {profile.insights.map((insight, i) => (
                    <p key={i} className="text-[10px] leading-snug text-[var(--color-text-light)]">
                      <span className="mr-1 opacity-40">&#x25B8;</span>{insight}
                    </p>
                  ))}
                </div>
              </div>
            );
          };

          return (
            <div className={`mt-3 grid gap-2 ${hasCompareMedia ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
              {renderProportionCard(primaryDisplay, primaryProps, primaryShowLive)}
              {hasCompareMedia && compareDisplay && compareProps
                ? renderProportionCard(compareDisplay, compareProps, compareShowLive)
                : null}
            </div>
          );
        })()}
        <div className="mt-2 flex flex-wrap items-center justify-end gap-2 border-t border-[var(--color-accent)]/10 pt-2 text-[10px]">
          <button
            type="button"
            onClick={() => setShowAngleGuide((v) => !v)}
            className="rounded-md border border-[var(--color-accent)]/20 px-2 py-1 text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)] transition-colors"
          >
            {showAngleGuide ? 'Hide guide' : 'Show guide'}
          </button>
          {isSquat && (
            <button
              type="button"
              onClick={() => setShowBiomechanics((v) => !v)}
              className="rounded-md border border-[var(--color-accent)]/20 px-2 py-1 text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)] transition-colors"
            >
              {showBiomechanics ? 'Hide biomechanics' : 'Show biomechanics'}
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              const next: FacingDirection = primaryFacingDirection === 'right' ? 'left' : 'right';
              setPrimaryFacingDirection(next);
              setKneeTrackingSide(facingDirectionToLeg(next));
            }}
            className="flex items-center gap-0.5 rounded-md border border-[var(--color-accent)]/20 px-2 py-1 text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)] transition-colors"
            title={`Primary facing ${primaryFacingDirection} (auto-detected) — click to override`}
          >
            {primaryFacingDirection === 'right'
              ? <><span>Primary facing</span><ChevronRight className="h-3 w-3" /></>
              : <><ChevronLeft className="h-3 w-3" /><span>Primary facing</span></>
            }
          </button>
          {hasCompareMedia && (
            <button
              type="button"
              onClick={() => {
                const next: FacingDirection = compareFacingDirection === 'right' ? 'left' : 'right';
                setCompareFacingDirection(next);
                setCompareKneeTrackingSide(facingDirectionToLeg(next));
              }}
              className="flex items-center gap-0.5 rounded-md border border-[var(--color-accent)]/20 px-2 py-1 text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)] transition-colors"
              title={`Compare facing ${compareFacingDirection} (auto-detected) — click to override`}
            >
              {compareFacingDirection === 'right'
                ? <><span>Compare facing</span><ChevronRight className="h-3 w-3" /></>
                : <><ChevronLeft className="h-3 w-3" /><span>Compare facing</span></>
              }
            </button>
          )}
        </div>
      </section>
    );
  };

  const mediaMaxClass =
    'max-h-[min(88dvh,1400px)] md:max-h-[min(85dvh,1400px)] lg:max-h-[min(84dvh,1500px)]';

  const renderMeasurementOverlay = () => {
    const angleMs = measurements.filter(m => m.kind === 'angle' && m.angle !== null);
    const lineMs = measurements.filter(m => m.kind === 'line' && m.length !== null && m.points.length === 2);
    const nodes: ReactNode[] = [];
    measurements.forEach((m, i) => {
      if (m.kind === 'angle' && m.angle !== null) {
        const idx = angleMs.indexOf(m) + 1;
        nodes.push(
          <div key={`angle-${i}`} className="rounded-xl px-3 py-1.5 bg-black/50 backdrop-blur-md">
            <p className="text-[10px] text-white/60">{angleMs.length > 1 ? `Angle ${idx}` : 'Angle'}</p>
            <p className="text-xl font-bold text-[var(--color-accent)]">{m.angle.toFixed(1)}°</p>
          </div>,
        );
      }
      if (m.kind === 'line' && m.length !== null && m.points.length === 2) {
        const idx = lineMs.indexOf(m) + 1;
        nodes.push(
          <div key={`line-${i}`} className="rounded-xl px-3 py-1.5 bg-black/50 backdrop-blur-md">
            <p className="text-[10px] text-white/60">{lineMs.length > 1 ? `Line ${idx}` : 'Line'}</p>
            <p className="text-xl font-bold text-[var(--color-accent)]">{m.length.toFixed(0)} px</p>
          </div>,
        );
      }
    });
    if (nodes.length === 0) return null;
    return (
      <div className="pointer-events-none absolute left-3 top-3 z-20 flex flex-col gap-1">
        {nodes}
      </div>
    );
  };

  const renderOverlayToolbelt = () => (
    <div className="pointer-events-none absolute inset-y-0 right-0 z-20 flex items-center">
      <div className="pointer-events-auto flex max-h-full flex-col gap-1 overflow-y-auto rounded-2xl border border-[var(--color-accent)]/15 bg-black/35 p-1.5 backdrop-blur-md">
        <button
          type="button"
          onClick={() => setShowAnalysis((v) => !v)}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${showAnalysis ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
          title={showAnalysis ? 'Hide analysis panel' : 'Show analysis panel'}
          aria-label={showAnalysis ? 'Hide analysis panel' : 'Show analysis panel'}
        >
          <LineChart className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={handleDeleteMeasurements}
          disabled={measurements.length === 0}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${measurements.length === 0 ? 'opacity-50 cursor-not-allowed' : 'text-red-300'}`}
          title="Clear measurements"
          aria-label="Clear measurements"
        >
          <Trash className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={toggleAngleTool}
          disabled={!isMediaLoaded}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'angle' ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
          title="Angle measure (three points)"
          aria-label="Angle measure"
        >
          <AngleMeasureIcon className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={toggleLineTool}
          disabled={!isMediaLoaded}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'line' ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
          title="Line measure (distance between two points)"
          aria-label="Line measure"
        >
          <Minus className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={() => setPoseEnabled((v) => !v)}
          disabled={!isMediaLoaded}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${poseEnabled ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
          title="Toggle pose overlay"
          aria-label="Toggle pose overlay"
        >
          <Activity className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={() => zoomByButton(ZOOM_BUTTON_FACTOR)}
          disabled={!isMediaLoaded}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-fg`}
          title="Zoom in"
          aria-label="Zoom in"
        >
          <ZoomIn className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={() => zoomByButton(1 / ZOOM_BUTTON_FACTOR)}
          disabled={!isMediaLoaded}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-fg`}
          title="Zoom out"
          aria-label="Zoom out"
        >
          <ZoomOut className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={togglePanTool}
          disabled={!isMediaLoaded || activeScale <= ZOOM_MIN_SCALE}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded || activeScale <= ZOOM_MIN_SCALE ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'pan' ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
          title="Pan tool"
          aria-label="Pan tool"
        >
          <Hand className="w-5 h-5" />
        </button>
        <button
          type="button"
          onClick={() => void toggleFullscreen()}
          disabled={!isMediaLoaded}
          className={`shrink-0 p-1.5 rounded-lg hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : 'text-fg'}`}
          title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
          aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
        >
          {isFullscreen ? <Minimize className="w-5 h-5" /> : <Maximize className="w-5 h-5" />}
        </button>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-[var(--color-bg-dark)] text-fg font-sans blueprint-bg">
      <header className="mb-5 flex items-center justify-between gap-4 border-b border-[var(--color-accent)]/20 bg-[var(--color-chrome-bar)] px-4 py-4 shadow-md md:px-8">
        <h1 className="min-w-0 text-2xl font-semibold leading-tight tracking-tight text-[var(--color-accent)] brand-font">
          Form Analyzer
        </h1>
        <SettingsMenu />
      </header>

      <main className="grid gap-6 px-4 pt-3 pb-12 md:px-8 md:pt-5">
        <section className="glass p-6 rounded-2xl border border-[var(--color-accent)]/10 shadow-lg accent-glow">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-fg brand-font">Media Analysis</h2>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={openGalleryPicker}
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[var(--color-accent)]/20"
                title="Pick image or video from gallery"
                aria-label="Pick image or video from gallery"
              >
                <ImageGalleryIcon className="w-6 h-6 text-[var(--color-accent)]" />
              </button>
              <button
                type="button"
                onClick={openComparePicker}
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[var(--color-accent)]/20"
                title="Add compare media"
                aria-label="Add compare media"
              >
                <Images className="w-6 h-6 text-[var(--color-accent)]" />
              </button>
              {hasCompareMedia ? (
                <button
                  type="button"
                  onClick={() => {
                    setCompareVideoSrc(null);
                    setCompareImageSrc(null);
                    setCompareAppliedZoom(null);
                    setIsComparePanning(false);
                  }}
                  className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[var(--color-accent)]/20 text-red-400"
                  title="Remove compare media"
                  aria-label="Remove compare media"
                >
                  <Trash className="w-6 h-6" />
                </button>
              ) : null}
              <input
                ref={galleryPickInputRef}
                type="file"
                accept="image/*,video/mp4,video/webm,video/ogg,video/quicktime,video/*"
                className="sr-only"
                aria-hidden
                tabIndex={-1}
                onChange={handleFileUpload}
              />
              <input
                ref={comparePickInputRef}
                type="file"
                accept="image/*,video/mp4,video/webm,video/ogg,video/quicktime,video/*"
                className="sr-only"
                aria-hidden
                tabIndex={-1}
                onChange={handleCompareFileUpload}
              />
              <input
                ref={imagePickInputRef}
                type="file"
                accept="image/*"
                className="sr-only"
                aria-hidden
                tabIndex={-1}
                onChange={handleFileUpload}
              />
              <input
                ref={videoPickInputRef}
                type="file"
                accept="video/mp4,video/webm,video/ogg,video/quicktime,video/*"
                className="sr-only"
                aria-hidden
                tabIndex={-1}
                onChange={handleFileUpload}
              />
              <button
                type="button"
                onClick={() => void openDeviceCamera()}
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[var(--color-accent)]/20"
                title="Take a photo"
                aria-label="Take a photo with the camera"
              >
                <Camera className="w-6 h-6 text-[var(--color-accent)]" />
              </button>
              <button
                type="button"
                onClick={() => void openDeviceVideoRecord()}
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[var(--color-accent)]/20"
                title="Record video with the camera"
                aria-label="Record video with the camera"
              >
                <Video className="w-6 h-6 text-[var(--color-accent)]" />
              </button>
            </div>
          </div>

          {videoSrc ? (
            <div
              ref={fullscreenTargetRef}
              className={`flex w-full min-w-0 flex-col ${isFullscreen ? 'relative h-[100dvh] rounded-none' : 'gap-3'}`}
            >
              {/* Tall phone media: no fixed 16:9 box; cap desktop height so the page fits the window */}
              <div
                className={`relative flex w-full min-w-0 items-center justify-center overflow-hidden ${isFullscreen ? 'h-full w-full rounded-none border-0 p-0' : 'min-h-[min(60dvh,640px)] rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-0'}`}
              >
                {renderOverlayToolbelt()}
                {renderMeasurementOverlay()}
                <div
                  className={`grid w-full min-w-0 items-stretch justify-items-stretch ${hasCompareMedia ? 'md:grid-cols-2 md:gap-0' : 'grid-cols-1 gap-0'}`}
                >
                  {/* Side-by-side: align portrait media to the inner seam (object-contain letterboxing otherwise sits in the middle). */}
                  <div
                    className={`flex min-w-0 flex-col gap-2 ${hasCompareMedia ? 'md:items-end' : 'items-center'}`}
                  >
                    <div
                      ref={mediaWrapRef}
                      className={`relative inline-block min-w-0 max-w-full overflow-hidden ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                    >
                      <div
                        className={
                          appliedZoom
                            ? 'relative inline-block'
                            : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`
                        }
                        style={
                          appliedZoom
                            ? {
                                transform: `translate(${-appliedZoom.x * appliedZoom.s}px, ${-appliedZoom.y * appliedZoom.s}px) scale(${appliedZoom.s})`,
                                transformOrigin: '0 0',
                              }
                            : undefined
                        }
                      >
                        <video
                          ref={videoRef}
                          src={videoSrc}
                          preload="auto"
                          playsInline
                          className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                        />
                      </div>
                      <canvas
                        ref={canvasRef}
                        className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                        style={{
                          touchAction: 'none',
                          cursor: isPanning ? 'grabbing' : activeTool === 'pan' ? 'grab' : 'crosshair',
                        }}
                        onPointerDown={handleCanvasPointerDown}
                        onPointerMove={handleCanvasPointerMove}
                        onPointerUp={handleCanvasPointerUp}
                        onPointerCancel={handleCanvasPointerUp}
                        onPointerLeave={handleCanvasPointerUp}
                      />
                    </div>
                  </div>
                  {hasCompareMedia ? (
                    <div className="flex min-w-0 flex-col gap-2 md:items-start">
                      <div
                        ref={compareMediaWrapRef}
                        className="relative inline-block min-w-0 max-w-full overflow-hidden"
                      >
                        <div
                          className={
                            compareAppliedZoom
                              ? 'relative inline-block'
                              : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`
                          }
                          style={
                            compareAppliedZoom
                              ? {
                                  transform: `translate(${-compareAppliedZoom.x * compareAppliedZoom.s}px, ${-compareAppliedZoom.y * compareAppliedZoom.s}px) scale(${compareAppliedZoom.s})`,
                                  transformOrigin: '0 0',
                                }
                              : undefined
                          }
                        >
                          {compareVideoSrc ? (
                            <video
                              ref={compareVideoRef}
                              src={compareVideoSrc}
                              preload="auto"
                              playsInline
                              className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                            />
                          ) : compareImageSrc ? (
                            <img
                              ref={compareImageRef}
                              src={compareImageSrc}
                              className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                              alt="Compare media"
                            />
                          ) : null}
                        </div>
                        <canvas
                          ref={compareCanvasRef}
                          className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                          style={{
                            touchAction: 'none',
                            cursor: isComparePanning ? 'grabbing' : activeTool === 'pan' ? 'grab' : activeTool === 'angle' || activeTool === 'line' ? 'crosshair' : 'default',
                          }}
                          onPointerDown={handleCompareCanvasPointerDown}
                          onPointerMove={handleCompareCanvasPointerMove}
                          onPointerUp={handleCompareCanvasPointerUp}
                          onPointerCancel={handleCompareCanvasPointerUp}
                          onPointerLeave={handleCompareCanvasPointerUp}
                        />
                      </div>
                    </div>
                  ) : null}
                </div>

                {/* Playback controls overlay at bottom of media container — hidden in fullscreen when analysis is open */}
                {!(isFullscreen && showAnalysis) && (
                <div className={`pointer-events-none absolute inset-x-0 bottom-0 z-20 flex flex-col gap-1 px-3 pb-3 pt-8 rounded-b-xl ${isFullscreen ? 'bg-transparent' : 'bg-gradient-to-t from-black/50 to-transparent'}`}>
                  <div className="pointer-events-auto flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setIsPlaying((v) => !v)}
                      className="p-1.5 rounded-full hover:bg-white/15 text-white"
                      title={
                        compareVideoSrc
                          ? isPlaying
                            ? 'Pause both videos'
                            : 'Play both videos'
                          : isPlaying
                            ? 'Pause'
                            : 'Play'
                      }
                      aria-label={
                        compareVideoSrc
                          ? isPlaying
                            ? 'Pause both videos'
                            : 'Play both videos'
                          : isPlaying
                            ? 'Pause video'
                            : 'Play video'
                      }
                    >
                      {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                    </button>
                    <button
                      type="button"
                      onClick={() => stepBothFrames(-1)}
                      className="p-1.5 rounded-full hover:bg-white/15 text-white"
                      title="Step back one frame"
                      aria-label="Step back one frame"
                    >
                      <ChevronLeft className="w-5 h-5" />
                    </button>
                    <button
                      type="button"
                      onClick={() => stepBothFrames(1)}
                      className="p-1.5 rounded-full hover:bg-white/15 text-white"
                      title="Step forward one frame"
                      aria-label="Step forward one frame"
                    >
                      <ChevronRight className="w-5 h-5" />
                    </button>
                  </div>
                  <div className={`pointer-events-auto grid gap-2 ${compareVideoSrc ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                    <input
                      type="range"
                      min="0"
                      max={Math.max(duration, currentTime, 0.0001)}
                      value={Math.min(currentTime, Math.max(duration, currentTime, 0.0001))}
                      onChange={handleSeek}
                      step={SLIDER_SCRUB_STEP_SECONDS}
                      className="w-full accent-[var(--color-accent)]"
                      aria-label="Primary video position"
                    />
                    {compareVideoSrc ? (
                      <input
                        type="range"
                        min="0"
                        max={Math.max(compareDuration, compareCurrentTime, 0.0001)}
                        value={Math.min(
                          compareCurrentTime,
                          Math.max(compareDuration, compareCurrentTime, 0.0001),
                        )}
                        onChange={handleCompareSeek}
                        step={SLIDER_SCRUB_STEP_SECONDS}
                        className="w-full accent-[var(--color-accent)]"
                        aria-label="Compare video position"
                      />
                    ) : null}
                  </div>
                </div>
                )}
                {isFullscreen && showAnalysis && (
                  <div className="absolute inset-x-0 bottom-0 z-30 max-h-[50dvh] overflow-y-auto rounded-t-2xl bg-transparent p-4">
                    {renderKneeAnglePanel()}
                  </div>
                )}
              </div>
              {!isFullscreen && showAnalysis ? renderKneeAnglePanel() : null}
            </div>
          ) : imageSrc ? (
            <div
              ref={fullscreenTargetRef}
              className={`flex w-full min-w-0 flex-col ${isFullscreen ? 'relative h-[100dvh] rounded-none' : 'gap-3'}`}
            >
              <div
                className={`relative flex w-full min-w-0 items-center justify-center overflow-hidden ${isFullscreen ? 'h-full w-full rounded-none border-0 p-0' : 'min-h-[min(60dvh,640px)] rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-0'}`}
              >
                {renderOverlayToolbelt()}
                {renderMeasurementOverlay()}
                <div
                  className={`grid w-full min-w-0 items-stretch justify-items-stretch ${hasCompareMedia ? 'md:grid-cols-2 md:gap-0' : 'grid-cols-1 gap-0'}`}
                >
                  <div className={`min-w-0 ${hasCompareMedia ? 'md:flex md:flex-col md:items-end' : 'flex flex-col items-center'}`}>
                    <div
                      ref={mediaWrapRef}
                      className={`relative inline-block min-w-0 max-w-full overflow-hidden ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                    >
                    <div
                      className={
                          compareAppliedZoom
                          ? 'relative inline-block'
                          : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`
                      }
                      style={
                          compareAppliedZoom
                          ? {
                                transform: `translate(${-compareAppliedZoom.x * compareAppliedZoom.s}px, ${-compareAppliedZoom.y * compareAppliedZoom.s}px) scale(${compareAppliedZoom.s})`,
                              transformOrigin: '0 0',
                            }
                          : undefined
                      }
                    >
                      <img
                        ref={imageRef}
                        src={imageSrc}
                        className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                        alt="Uploaded"
                        onLoad={() => { setMeasurements([]); setActiveMeasurementIdx(null); }}
                      />
                    </div>
                    <canvas
                      ref={canvasRef}
                      className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                      style={{
                        touchAction: 'none',
                        cursor: isPanning ? 'grabbing' : activeTool === 'pan' ? 'grab' : 'crosshair',
                      }}
                      onPointerDown={handleCanvasPointerDown}
                      onPointerMove={handleCanvasPointerMove}
                      onPointerUp={handleCanvasPointerUp}
                      onPointerCancel={handleCanvasPointerUp}
                      onPointerLeave={handleCanvasPointerUp}
                    />
                  </div>
                  </div>
                  {hasCompareMedia ? (
                    <div className={`min-w-0 ${hasCompareMedia ? 'md:flex md:flex-col md:items-start' : ''}`}>
                    <div
                      ref={compareMediaWrapRef}
                      className="relative inline-block min-w-0 max-w-full overflow-hidden"
                    >
                      <div
                        className={
                          appliedZoom
                            ? 'relative inline-block'
                            : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`
                        }
                        style={
                          appliedZoom
                            ? {
                                transform: `translate(${-appliedZoom.x * appliedZoom.s}px, ${-appliedZoom.y * appliedZoom.s}px) scale(${appliedZoom.s})`,
                                transformOrigin: '0 0',
                              }
                            : undefined
                        }
                      >
                        {compareVideoSrc ? (
                          <video
                            ref={compareVideoRef}
                            src={compareVideoSrc}
                            preload="auto"
                            playsInline
                            className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                          />
                        ) : compareImageSrc ? (
                          <img
                            ref={compareImageRef}
                            src={compareImageSrc}
                            className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : mediaMaxClass}`}
                            alt="Compare media"
                          />
                        ) : null}
                      </div>
                      <canvas
                        ref={compareCanvasRef}
                        className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                        style={{
                          touchAction: 'none',
                          cursor: isComparePanning ? 'grabbing' : activeTool === 'pan' ? 'grab' : activeTool === 'angle' || activeTool === 'line' ? 'crosshair' : 'default',
                        }}
                        onPointerDown={handleCompareCanvasPointerDown}
                        onPointerMove={handleCompareCanvasPointerMove}
                        onPointerUp={handleCompareCanvasPointerUp}
                        onPointerCancel={handleCompareCanvasPointerUp}
                        onPointerLeave={handleCompareCanvasPointerUp}
                      />
                    </div>
                    </div>
                  ) : null}
                </div>
                {isFullscreen && showAnalysis && (
                  <div className="absolute inset-x-0 bottom-0 z-30 max-h-[50dvh] overflow-y-auto rounded-t-2xl bg-transparent p-4">
                    {renderKneeAnglePanel()}
                  </div>
                )}
              </div>
              {!isFullscreen && showAnalysis ? renderKneeAnglePanel() : null}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 rounded-xl border-2 border-dashed border-[var(--color-accent)]/20 bg-[var(--color-bg-dark)] p-8 text-center">
              <div className="mb-4 flex gap-3 text-[var(--color-accent)]/50">
                <ImageGalleryIcon className="h-10 w-10" />
                <Camera className="h-10 w-10" />
                <Video className="h-10 w-10" />
              </div>
              <p className="text-[var(--color-text-light)] max-w-sm">
                Add media with the buttons above: gallery (images or videos), take a photo, or record video.
              </p>
            </div>
          )}
        </section>

      </main>

      {cameraModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="camera-dialog-title"
        >
          <div className="glass accent-glow w-full max-w-lg rounded-2xl border border-[var(--color-accent)]/10 p-4 shadow-xl">
            <h3 id="camera-dialog-title" className="mb-3 text-lg font-semibold text-fg brand-font">
              Camera
            </h3>
            <div className="overflow-hidden rounded-xl border border-[var(--color-accent)]/10 bg-black">
              <video
                ref={cameraPreviewRef}
                autoPlay
                playsInline
                muted
                className="max-h-[min(70dvh,520px)] w-full object-contain sm:max-h-[520px]"
              />
            </div>
            {cameraError ? (
              <p className="mt-2 text-sm text-red-300" role="alert">
                {cameraError}
              </p>
            ) : null}
            <div className="mt-4 flex flex-wrap justify-end gap-2">
              <button
                type="button"
                onClick={closeCameraModal}
                className="rounded-lg border border-[var(--color-accent)]/20 bg-[var(--color-bg-dark)] px-4 py-2 text-sm hover:bg-[var(--color-panel-hover)]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={capturePhotoFromCamera}
                className="rounded-lg border border-[var(--color-accent)]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-white hover:bg-[var(--color-accent-hover)]"
              >
                Capture photo
              </button>
            </div>
          </div>
        </div>
      )}

      {videoRecordModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="video-record-dialog-title"
        >
          <div className="glass accent-glow w-full max-w-lg rounded-2xl border border-[var(--color-accent)]/10 p-4 shadow-xl">
            <h3 id="video-record-dialog-title" className="mb-3 text-lg font-semibold text-fg brand-font">
              Record video
            </h3>
            <div className="overflow-hidden rounded-xl border border-[var(--color-accent)]/10 bg-black">
              <video
                ref={videoRecordPreviewRef}
                autoPlay
                playsInline
                muted
                className="max-h-[min(70dvh,520px)] w-full object-contain sm:max-h-[520px]"
              />
            </div>
            {isRecordingVideo ? (
              <p className="mt-2 text-sm text-[var(--color-text-light)]">Recording…</p>
            ) : null}
            {videoRecordError ? (
              <p className="mt-2 text-sm text-red-300" role="alert">
                {videoRecordError}
              </p>
            ) : null}
            <div className="mt-4 flex flex-wrap justify-end gap-2">
              <button
                type="button"
                onClick={closeVideoRecordModal}
                className="rounded-lg border border-[var(--color-accent)]/20 bg-[var(--color-bg-dark)] px-4 py-2 text-sm hover:bg-[var(--color-panel-hover)]"
              >
                Cancel
              </button>
              {!isRecordingVideo ? (
                <button
                  type="button"
                  onClick={startVideoRecording}
                  className="rounded-lg border border-[var(--color-accent)]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-white hover:bg-[var(--color-accent-hover)]"
                >
                  Start recording
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopVideoRecordingAndSave}
                  className="rounded-lg border border-[var(--color-accent)]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-white hover:bg-[var(--color-accent-hover)]"
                >
                  Stop and use clip
                </button>
              )}
            </div>
          </div>
        </div>
      )}
      {videoSrc && (
        <video
          ref={bgVideoRef}
          src={videoSrc}
          preload="auto"
          playsInline
          muted
          style={{ position: 'fixed', top: -9999, left: -9999, width: 1, height: 1, opacity: 0, pointerEvents: 'none' }}
        />
      )}
      {compareVideoSrc && (
        <video
          ref={compareBgVideoRef}
          src={compareVideoSrc}
          preload="auto"
          playsInline
          muted
          style={{ position: 'fixed', top: -9999, left: -9999, width: 1, height: 1, opacity: 0, pointerEvents: 'none' }}
        />
      )}
    </div>
  );
}


