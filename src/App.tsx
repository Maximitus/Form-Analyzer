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
  PointerEvent as ReactPointerEvent,
} from 'react';
import {
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Hand,
  Maximize,
  Minimize,
  Ruler,
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
const POSE_TRACKED_IDS = [11, 12, 13, 14, 15, 16, 5, 6, 7, 8, 9, 10];
const POSE_LEFT_SIDE_IDS = new Set([5, 7, 9, 11, 13, 15]);
const POSE_RIGHT_SIDE_IDS = new Set([6, 8, 10, 12, 14, 16]);
const POSE_ARM_IDS = new Set([7, 8, 9, 10]);
const POSE_BODY_CONNECTIONS: [number, number][] = [
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
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
const KNEE_MAXIMA_MIN_SAMPLES = 3;
const KNEE_MAXIMA_MIN_GAP_SECONDS = 0.1;
const KNEE_MINIMA_MIN_SAMPLES = 3;
const KNEE_MINIMA_MIN_GAP_SECONDS = 0.3;

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
  return Math.atan2(dy, dx) * (180 / Math.PI);
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
  },
): KneeAnglePeak[] {
  if (series.length < KNEE_MAXIMA_MIN_SAMPLES) return [];
  const candidates: {index: number; angle: number; direction: ExtensionDirection}[] = [];
  for (let i = 1; i < series.length - 1; i += 1) {
    const prev = series[i - 1]!.angle;
    const current = series[i]!.angle;
    const next = series[i + 1]!.angle;
    const isPeak = (current >= prev && current > next) || (current > prev && current >= next);
    if (!isPeak || current < options.lowerAngle || current > options.upperAngle) continue;
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

function detectKneeAngleMinima(
  series: {time: number; angle: number; depthAngle: number | null; hipAngle: number | null; backAngle: number | null}[],
  options: {
    lowerAngle: number;
    upperAngle: number;
  },
): KneeAngleValley[] {
  if (series.length < KNEE_MINIMA_MIN_SAMPLES) return [];
  const candidates: {index: number; angle: number}[] = [];
  for (let i = 1; i < series.length - 1; i += 1) {
    const prev = series[i - 1]!.angle;
    const current = series[i]!.angle;
    const next = series[i + 1]!.angle;
    const isValley = (current <= prev && current < next) || (current < prev && current <= next);
    if (!isValley || current < options.lowerAngle || current > options.upperAngle) continue;
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
  const [angle, setAngle] = useState<number | null>(null);
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [poseEnabled, setPoseEnabled] = useState(false);
  const [poseKeypoints, setPoseKeypoints] = useState<{x: number; y: number; v: number}[]>([]);
  const [comparePoseKeypoints, setComparePoseKeypoints] = useState<{x: number; y: number; v: number}[]>([]);
  const [currentKneeAngle, setCurrentKneeAngle] = useState<number | null>(null);
  const [currentLiveDepthAngle, setCurrentLiveDepthAngle] = useState<number | null>(null);
  const [currentLiveHipAngle, setCurrentLiveHipAngle] = useState<number | null>(null);
  const [currentLiveBackAngle, setCurrentLiveBackAngle] = useState<number | null>(null);
  const [kneeAngleSeries, setKneeAngleSeries] = useState<{time: number; angle: number; hipX: number; ankleX: number; depthAngle: number | null; hipAngle: number | null; backAngle: number | null}[]>([]);
  const [posePointSeries, setPosePointSeries] = useState<PoseFrameSample[]>([]);
  const [kneeTrackingSide, setKneeTrackingSide] = useState<KneeSide | null>(null);
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
  const [extensionFilter, setExtensionFilter] = useState<ExtensionDirection>('forward');
  const [facingDirection, setFacingDirection] = useState<FacingDirection>('right');
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('stride');
  const poseDetectorRef = useRef<poseDetection.PoseDetector | null>(null);
  const poseRafRef = useRef<number | null>(null);
  const bgVideoRef = useRef<HTMLVideoElement | null>(null);
  const [poseStatus, setPoseStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
  const poseCacheRef = useRef<{time: number; keypoints: {x: number; y: number; v: number}[]}[]>([]);
  const [analysisProgress, setAnalysisProgress] = useState<number | null>(null);
  const analysisAbortRef = useRef(false);
  const analysisGenRef = useRef(0);
  const [compareMediaLayoutVersion, setCompareMediaLayoutVersion] = useState(0);

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

  /** `ruler` = angle tool, `pan` = drag viewport while zoomed */
  type AnalysisTool = 'ruler' | 'pan' | null;
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
      setPoints([]);
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
        setPoints([]);
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
        setPoints([]);
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
        poseDetection.SupportedModels.MoveNet,
        {
          modelType: poseDetection.movenet.modelType.SINGLEPOSE_THUNDER,
          enableSmoothing: true,
        },
      );
      poseDetectorRef.current = detector;
      setPoseStatus('ready');
      return detector;
    };

    const updatePoseFromResult = (
      poses: poseDetection.Pose[] | null | undefined,
      source: HTMLVideoElement | HTMLImageElement | null,
      setter: (next: {x: number; y: number; v: number}[]) => void,
      autoDetectFacing = false,
    ) => {
      const keypoints = poses?.[0]?.keypoints;
      const sourceWidth =
        source instanceof HTMLVideoElement ? source.videoWidth : source instanceof HTMLImageElement ? source.naturalWidth : 0;
      const sourceHeight =
        source instanceof HTMLVideoElement ? source.videoHeight : source instanceof HTMLImageElement ? source.naturalHeight : 0;
      if (!keypoints || keypoints.length === 0 || sourceWidth <= 0 || sourceHeight <= 0) {
        setter([]);
        return;
      }
      const selected = POSE_TRACKED_IDS.map((id) => {
        const p = keypoints[id];
        if (!p) return {x: 0, y: 0, v: 0};
        const score = p.score ?? 0;
        const coords = p as {x: number; y: number};
        return {x: coords.x / sourceWidth, y: coords.y / sourceHeight, v: score};
      });
      setter(selected);
      if (autoDetectFacing) {
        const detected = detectFacingFromKeypoints(selected);
        if (detected) {
          setFacingDirection(detected);
          setKneeTrackingSide(facingDirectionToLeg(detected));
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
            if (!cancelled) updatePoseFromResult(result, primaryImage, setPoseKeypoints, true);
          }
          if (compareImageSrc && compareImage) {
            const result = await detector.estimatePoses(compareImage, {flipHorizontal: false});
            if (!cancelled) updatePoseFromResult(result, compareImage, setComparePoseKeypoints);
          }
          return;
        }

        if (imageSrc && primaryImage) {
          // Primary is an image: detect once and keep it.
          const result = await detector.estimatePoses(primaryImage, {flipHorizontal: false});
          if (!cancelled) updatePoseFromResult(result, primaryImage, setPoseKeypoints);
        }
        if (compareImageSrc && compareImage) {
          const result = await detector.estimatePoses(compareImage, {flipHorizontal: false});
          if (!cancelled) updatePoseFromResult(result, compareImage, setComparePoseKeypoints);
        }

        const tick = async () => {
          if (cancelled) return;
          const now = performance.now();

          if (videoSrc && primaryVideo) {
            const v = primaryVideo;
            if (
              v.readyState >= 2 &&
              (v.currentTime !== lastPrimaryVideoTime || !v.paused) &&
              now - lastPrimaryInferenceTs >= minInferenceIntervalMs
            ) {
              lastPrimaryVideoTime = v.currentTime;
              lastPrimaryInferenceTs = now;
              const result = await detector.estimatePoses(v, {flipHorizontal: false});
              if (!cancelled) updatePoseFromResult(result, v, setPoseKeypoints);
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
              const result = await detector.estimatePoses(v, {flipHorizontal: false});
              if (!cancelled) updatePoseFromResult(result, v, setComparePoseKeypoints);
            }
          }

          poseRafRef.current = requestAnimationFrame(tick);
        };

        tick();
      } catch {
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

      if (points.length > 0) {
        points.forEach(p => {
          const o = contentToOverlay(p.x, p.y, zMap);
          ctx.beginPath();
          ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
          ctx.fill();
        });
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

        for (const [aId, bId] of POSE_BODY_CONNECTIONS) {
          const a = byId.get(aId);
          const b = byId.get(bId);
          if (!a || !b || a.v < 0.25 || b.v < 0.25) continue;
          const arm = isArmConnection(aId, bId);
          const tracked = !arm && isTrackedConnection(aId, bId);
          ctx.strokeStyle = arm ? 'rgba(0,210,255,0.15)' : tracked ? '#00d2ff' : 'rgba(0,210,255,0.25)';
          ctx.lineWidth = arm ? 1 : tracked ? 3 : 1.5;
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
          const tracked = !arm && trackedIds.has(id);
          ctx.fillStyle = arm ? 'rgba(0,210,255,0.15)' : tracked ? '#00d2ff' : 'rgba(0,210,255,0.25)';
          const o = mapPoseNormToCanvasOverlayPx(p.x, p.y, videoRef.current ?? imageRef.current, canvasRef.current);
          ctx.beginPath();
          ctx.arc(o.x, o.y, arm ? 2.5 : tracked ? 5 : 3.5, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.restore();
      }

      if (points.length >= 2) {
        const o0 = contentToOverlay(points[0].x, points[0].y, zMap);
        const o1 = contentToOverlay(points[1].x, points[1].y, zMap);
        ctx.beginPath();
        ctx.moveTo(o0.x, o0.y);
        ctx.lineTo(o1.x, o1.y);
        if (points.length === 3) {
          const o2 = contentToOverlay(points[2].x, points[2].y, zMap);
          ctx.lineTo(o2.x, o2.y);

          ctx.font = '20px Space Grotesk';
          ctx.fillStyle = accent;
          ctx.fillText(`${angle?.toFixed(1)}°`, o1.x + 10, o1.y - 10);
        }
        ctx.stroke();
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
  }, [points, videoSrc, imageSrc, angle, mediaLayoutVersion, appliedZoom, poseEnabled, poseKeypoints, accentId, kneeTrackingSide, currentTime]);

  // Compare panel pose overlay (independent from primary zoom/pan)
  useEffect(() => {
    const canvas = compareCanvasRef.current;
    const video = compareVideoRef.current;
    const image = compareImageRef.current;
    if (!canvas || (!video && !image)) return;

    const wrap = canvas.parentElement;
    if (!wrap) return;

    const vw = wrap.clientWidth;
    const vh = wrap.clientHeight;
    canvas.width = Math.max(1, Math.round(vw));
    canvas.height = Math.max(1, Math.round(vh));

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!poseEnabled) return;
    if (comparePoseKeypoints.length !== POSE_TRACKED_IDS.length) return;

    const media = (video ?? image) as HTMLVideoElement | HTMLImageElement;

    const byId = new Map<number, {x: number; y: number; v: number}>();
    POSE_TRACKED_IDS.forEach((id, i) => {
      const p = comparePoseKeypoints[i];
      if (p) byId.set(id, p);
    });

    ctx.save();
    ctx.strokeStyle = '#00d2ff';
    ctx.fillStyle = '#00d2ff';
    ctx.lineWidth = 2.5;

    for (const [aId, bId] of POSE_BODY_CONNECTIONS) {
      const a = byId.get(aId);
      const b = byId.get(bId);
      if (!a || !b || a.v < 0.25 || b.v < 0.25) continue;
      const aPx = mapPoseNormToCanvasOverlayPx(a.x, a.y, media, canvas);
      const bPx = mapPoseNormToCanvasOverlayPx(b.x, b.y, media, canvas);
      ctx.beginPath();
      ctx.moveTo(aPx.x, aPx.y);
      ctx.lineTo(bPx.x, bPx.y);
      ctx.stroke();
    }

    for (const p of comparePoseKeypoints) {
      if (p.v < 0.25) continue;
      const o = mapPoseNormToCanvasOverlayPx(p.x, p.y, media, canvas);
      ctx.beginPath();
      ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }, [comparePoseKeypoints, compareVideoSrc, compareImageSrc, compareMediaLayoutVersion, poseEnabled]);

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

    let activeSide = facingDirectionToLeg(facingDirection);
    let detectedFacing: FacingDirection | null = null;

    analysisAbortRef.current = true;
    const gen = ++analysisGenRef.current;
    const isStale = () => analysisGenRef.current !== gen;

    setKneeAngleSeries([]);
    setPosePointSeries([]);
    poseCacheRef.current = [];
    setKneeTrackingSide(activeSide);
    setIsKneeGraphLocked(false);
    setIsPoseAnalyzing(true);
    setGraphDomainTime(targetDuration);
    setAnalysisProgress(0);
    analysisAbortRef.current = false;

    const ANALYSIS_FPS = 30;
    const frameStep = 1 / ANALYSIS_FPS;
    const totalFrames = Math.ceil(targetDuration * ANALYSIS_FPS);

    const PROGRESSIVE_BATCH = 15;

    const waitForBgVideoReady = (): Promise<boolean> =>
      new Promise((resolve) => {
        if (bgVideo.readyState >= 2) { resolve(true); return; }
        let resolved = false;
        const finish = (ok: boolean) => {
          if (resolved) return;
          resolved = true;
          bgVideo.removeEventListener('canplay', onReady);
          resolve(ok);
        };
        const onReady = () => finish(true);
        bgVideo.addEventListener('canplay', onReady);
        setTimeout(() => finish(bgVideo.readyState >= 2), 5000);
      });

    const ready = await waitForBgVideoReady();
    if (!ready || isStale()) {
      if (!isStale()) {
        setIsPoseAnalyzing(false);
        setAnalysisProgress(null);
      }
      return;
    }

    const sourceWidth = bgVideo.videoWidth || mainVideo.videoWidth;
    const sourceHeight = bgVideo.videoHeight || mainVideo.videoHeight;
    const ar = sourceHeight > 0 ? sourceWidth / sourceHeight : 1;

    const seekAndWait = (t: number): Promise<void> =>
      new Promise((resolve) => {
        const clamped = Math.min(t, targetDuration);
        if (Math.abs(bgVideo.currentTime - clamped) < 0.005) {
          resolve();
          return;
        }
        let resolved = false;
        const finish = () => {
          if (resolved) return;
          resolved = true;
          bgVideo.removeEventListener('seeked', finish);
          resolve();
        };
        bgVideo.addEventListener('seeked', finish);
        bgVideo.currentTime = clamped;
        setTimeout(finish, 2000);
      });

    const yieldToUI = () => new Promise<void>((r) => setTimeout(r, 0));

    const angleSeries: {time: number; angle: number; hipX: number; ankleX: number; depthAngle: number | null; hipAngle: number | null; backAngle: number | null}[] = [];
    const pointSeries: PoseFrameSample[] = [];
    const cache: {time: number; keypoints: {x: number; y: number; v: number}[]}[] = [];
    let t = 0;
    let frameIdx = 0;

    while (t <= targetDuration && !isStale()) {
      await seekAndWait(t);
      if (isStale()) break;
      if (bgVideo.readyState < 2) {
        await new Promise<void>((r) => {
          const onReady = () => { bgVideo.removeEventListener('canplay', onReady); r(); };
          bgVideo.addEventListener('canplay', onReady);
          setTimeout(() => { bgVideo.removeEventListener('canplay', onReady); r(); }, 3000);
        });
      }

      const result = await detector.estimatePoses(bgVideo, {flipHorizontal: false});
      if (isStale()) break;
      const keypoints = result?.[0]?.keypoints;
      let normalized: {x: number; y: number; v: number}[] = [];
      if (keypoints && keypoints.length > 0 && sourceWidth > 0 && sourceHeight > 0) {
        normalized = POSE_TRACKED_IDS.map((id) => {
          const p = keypoints[id];
          if (!p) return {x: 0, y: 0, v: 0};
          return {x: p.x / sourceWidth, y: p.y / sourceHeight, v: p.score ?? 0};
        });
      }

      cache.push({time: t, keypoints: normalized});

      if (!detectedFacing && normalized.length === POSE_TRACKED_IDS.length) {
        const detected = detectFacingFromKeypoints(normalized);
        if (detected) {
          detectedFacing = detected;
          activeSide = facingDirectionToLeg(detected);
          if (!isStale()) {
            setFacingDirection(detected);
            setKneeTrackingSide(activeSide);
          }
        }
      }

      if (normalized.length === POSE_TRACKED_IDS.length) {
        const kneeAngle = getKneeAngleFromPose(normalized, activeSide, MIN_POSE_VISIBILITY, ar);
        const anchors = getTrackedLegAnchors(normalized, activeSide);
        const depth = getSquatDepthAngle(normalized, activeSide, MIN_POSE_VISIBILITY, ar);
        const hipJointAngle = getHipAngle(normalized, activeSide, MIN_POSE_VISIBILITY, ar);
        const torsoAngle = getBackAngle(normalized, activeSide, MIN_POSE_VISIBILITY, ar);
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
          pointSeries.push({
            time: t,
            side: activeSide,
            kneeAngle,
            hipX: anchors.hipX,
            ankleX: anchors.ankleX,
            extension,
            keypoints: normalized.map((p) => ({x: p.x, y: p.y, v: p.v})),
          });
        }
      }

      frameIdx += 1;
      const pct = Math.min(100, Math.round((frameIdx / totalFrames) * 100));
      setAnalysisProgress(pct);

      if (frameIdx % PROGRESSIVE_BATCH === 0) {
        poseCacheRef.current = cache.slice();
        setKneeAngleSeries(angleSeries.slice());
        setPosePointSeries(pointSeries.slice());
        await yieldToUI();
      } else if (frameIdx % 2 === 0) {
        await yieldToUI();
      }

      t += frameStep;
    }

    if (!isStale()) {
      poseCacheRef.current = cache;
      setKneeAngleSeries(angleSeries);
      setPosePointSeries(pointSeries);
      setIsPoseAnalyzing(false);
      setIsKneeGraphLocked(true);
      setAnalysisProgress(null);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingDirection]);

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
      poseCacheRef.current = [];
      setKneeTrackingSide(null);
      setIsKneeGraphLocked(false);
      setIsPoseAnalyzing(false);
      setAnalysisProgress(null);
      analysisAbortRef.current = true;
      return;
    }
    if (!videoSrc) {
      setKneeAngleSeries([]);
      setPosePointSeries([]);
      poseCacheRef.current = [];
    }
    if (!videoSrc && !imageSrc) {
      setKneeTrackingSide(null);
    }
  }, [poseEnabled, videoSrc, imageSrc]);

  useEffect(() => {
    setCurrentKneeAngle(null);
    setCurrentLiveDepthAngle(null);
    setCurrentLiveHipAngle(null);
    setCurrentLiveBackAngle(null);
    setKneeAngleSeries([]);
    setPosePointSeries([]);
    poseCacheRef.current = [];
    setKneeTrackingSide(null);
    setGraphDomainTime(0);
    setAnalysisProgress(null);
    analysisAbortRef.current = true;
    setIsPoseAnalyzing(false);
  }, [videoSrc, imageSrc]);

  useEffect(() => {
    if (!poseEnabled) {
      setCurrentKneeAngle(null);
      setCurrentLiveDepthAngle(null);
      setCurrentLiveHipAngle(null);
      setCurrentLiveBackAngle(null);
      return;
    }
    const sideFromFacing = facingDirectionToLeg(facingDirection);
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
  }, [poseEnabled, poseKeypoints, currentTime, videoSrc, kneeTrackingSide, facingDirection]);

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
    setPoints([]);
    setAngle(null);
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

  const getPointAtOverlay = (ox: number, oy: number) => {
    const zMap: AppliedZoomMap | null = appliedZoom
      ? {x: appliedZoom.x, y: appliedZoom.y, s: appliedZoom.s}
      : null;
    return points.findIndex(p => {
      const o = contentToOverlay(p.x, p.y, zMap);
      return Math.hypot(o.x - ox, o.y - oy) < 15;
    });
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

  const toggleRulerTool = () => {
    setActiveTool((prev) => {
      if (prev === 'ruler') {
        setIsDrawing(false);
        return null;
      }
      setIsDrawing(true);
      return 'ruler';
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

    const pointIndex = getPointAtOverlay(ox, oy);
    if (pointIndex !== -1) {
      e.currentTarget.setPointerCapture(e.pointerId);
      setDraggingPointIndex(pointIndex);
      return;
    }

    if (!isDrawing || activeTool !== 'ruler') return;

    /* Keep a completed 3-point angle until trash clears it — ignore extra taps (e.g. mobile pan). */
    if (points.length >= 3) return;

    e.currentTarget.setPointerCapture(e.pointerId);
    setPoints(prev => {
      const newPoints = [...prev, { x, y }];
      if (newPoints.length === 3) {
        calculateAngle(newPoints);
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
      if (newPoints.length === 3) {
        calculateAngle(newPoints);
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
    }
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
    }
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
  };

  const calculateAngle = (pts: { x: number; y: number }[]) => {
    const [p1, p2, p3] = pts;
    setAngle(calculateJointAngle(p1, p2, p3));
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

    const allMaxima = isSquat ? [] : detectKneeAngleMaxima(series, {
      lowerAngle: angleLowerBound,
      upperAngle: angleUpperBound,
      facing: facingDirection,
    });
    const maxima = allMaxima.filter((p) => p.direction === extensionFilter);
    const otherMaxima = allMaxima.filter((p) => p.direction !== extensionFilter);
    const avgPeakAngle =
      maxima.length > 0
        ? maxima.reduce((sum, peak) => sum + peak.angle, 0) / maxima.length
        : null;
    const maxPeakAngle =
      maxima.length > 0
        ? Math.max(...maxima.map((p) => p.angle))
        : null;

    const valleys = isSquat ? detectKneeAngleMinima(series, {
      lowerAngle: angleLowerBound,
      upperAngle: angleUpperBound,
    }) : [];
    const belowParallelCount = valleys.filter((v) => v.belowParallel).length;
    const avgRepKneeAngle =
      valleys.length > 0
        ? valleys.reduce((sum, v) => sum + v.angle, 0) / valleys.length
        : null;
    const avgDepthAngle =
      valleys.length > 0
        ? valleys.reduce((sum, v) => sum + v.depthAngle, 0) / valleys.length
        : null;
    const avgRepHipAngle = (() => {
      const withHip = valleys.filter((v) => v.hipAngle !== null);
      if (withHip.length === 0) return null;
      return withHip.reduce((sum, v) => sum + v.hipAngle!, 0) / withHip.length;
    })();
    const avgRepBackAngle = (() => {
      const withBack = valleys.filter((v) => v.backAngle !== null);
      if (withBack.length === 0) return null;
      return withBack.reduce((sum, v) => sum + v.backAngle!, 0) / withBack.length;
    })();
    const currentNearest = (() => {
      if (!isSquat || series.length === 0) return null;
      return series.reduce((best, s) =>
        Math.abs(s.time - currentTime) < Math.abs(best.time - currentTime) ? s : best
      );
    })();
    const currentDepthAngle = currentLiveDepthAngle ?? currentNearest?.depthAngle ?? null;
    const currentHipAngle = currentLiveHipAngle ?? currentNearest?.hipAngle ?? null;
    const currentBackAngle = currentLiveBackAngle ?? currentNearest?.backAngle ?? null;

    const seriesEndTime = series.length > 0 ? series[series.length - 1]!.time : 0;
    const sliderTime = Math.max(duration, currentTime, 0.0001);
    const maxTime = graphDomainTime > 0 ? graphDomainTime : Math.max(sliderTime, seriesEndTime, 0.0001);
    const graphHeight = 140;

    const strideMinAngle = 0;
    const strideMaxAngle = 180;

    const depthAngles = isSquat ? series.map(s => s.depthAngle).filter((d): d is number => d !== null) : [];
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

    const playheadX = maxTime > 0 ? (currentTime / maxTime) * 100 : 0;

    const linePath = series
      .map((sample, idx) => {
        const x = (sample.time / maxTime) * 100;
        const val = isSquat ? (sample.depthAngle ?? 0) : sample.angle;
        const y = graphHeight - ((val - gMin) / (gMax - gMin)) * graphHeight;
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(3)} ${y.toFixed(3)}`;
      })
      .join(' ');

    return (
      <section className="rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-3 sm:p-4">
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
          <div className="relative rounded-lg border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/70">
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
              </>
            ) : (
              <p className="py-8 text-center text-sm text-[var(--color-text-light)]">
                {isSquat
                  ? 'Play or scrub the video to collect squat-depth samples.'
                  : 'Play or scrub the video to collect knee-angle samples.'}
              </p>
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
          <>
            <div className="mt-2 grid grid-cols-4 gap-x-3 text-xs text-[var(--color-text-light)]">
              <p className="font-medium text-[var(--color-accent)]">Knee: {currentKneeAngle !== null ? `${currentKneeAngle.toFixed(1)}°` : '—'}</p>
              <p className="font-medium text-[var(--color-accent)]">Depth: {currentDepthAngle !== null ? `${currentDepthAngle.toFixed(1)}°` : '—'}</p>
              <p className="font-medium text-[var(--color-accent)]">Hip: {currentHipAngle !== null ? `${currentHipAngle.toFixed(1)}°` : '—'}</p>
              <p className="font-medium text-[var(--color-accent)]">Back: {currentBackAngle !== null ? `${currentBackAngle.toFixed(1)}°` : '—'}</p>
            </div>
            <div className="mt-1 grid grid-cols-4 gap-x-3 text-xs text-[var(--color-text-light)]">
              <p>Avg: {avgRepKneeAngle !== null ? `${avgRepKneeAngle.toFixed(1)}°` : '—'}</p>
              <p>Avg: {avgDepthAngle !== null ? `${avgDepthAngle.toFixed(1)}°` : '—'}</p>
              <p>Avg: {avgRepHipAngle !== null ? `${avgRepHipAngle.toFixed(1)}°` : '—'}</p>
              <p>Avg: {avgRepBackAngle !== null ? `${avgRepBackAngle.toFixed(1)}°` : '—'}</p>
            </div>
          </>
        
        ) : (
          <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-[var(--color-text-light)] sm:grid-cols-4">
            <p>Current: {currentKneeAngle !== null ? `${currentKneeAngle.toFixed(1)}°` : '—'}</p>
            <p>Samples: {series.length}</p>
            <p>{extensionFilter === 'forward' ? 'Forward' : 'Behind'} peaks: {maxima.length}</p>
            <p>
              Avg peak: {avgPeakAngle !== null ? `${avgPeakAngle.toFixed(1)}°` : '—'}
            </p>
            <p>
              Max peak: {maxPeakAngle !== null ? `${maxPeakAngle.toFixed(1)}°` : '—'}
            </p>
            <p>
              Bounds: {angleLowerBound.toFixed(0)}° – {angleUpperBound.toFixed(0)}°
            </p>
            <p className="flex items-center justify-end">
              <button
                type="button"
                onClick={() => {
                  const next: FacingDirection = facingDirection === 'right' ? 'left' : 'right';
                  setFacingDirection(next);
                  setKneeTrackingSide(facingDirectionToLeg(next));
                }}
                className="flex items-center gap-0.5 text-[10px] opacity-40 hover:opacity-70 transition-opacity text-[var(--color-text-light)]"
                title={`Facing ${facingDirection} (auto-detected) — click to override`}
              >
                {facingDirection === 'right'
                  ? <><span>Facing</span><ChevronRight className="h-3 w-3" /></>
                  : <><ChevronLeft className="h-3 w-3" /><span>Facing</span></>
                }
              </button>
            </p>
          </div>
        )}
        {isSquat && (
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[var(--color-text-light)]">
            <span className="text-xs">Reps: {valleys.length}</span>
            <span className="text-xs" style={{color: valleys.length > 0 ? (belowParallelCount === valleys.length ? '#52c41a' : belowParallelCount > 0 ? '#faad14' : '#ff4d4f') : undefined}}>
              Below parallel: {belowParallelCount}/{valleys.length}
            </span>
            {series.length >= 2 && (
              <>
                <span className="flex items-center gap-1"><span className="inline-block h-2 w-4 rounded-sm" style={{background: '#52c41a'}} /> below (depth &lt; 0°)</span>
                <span className="flex items-center gap-1"><span className="inline-block h-2 w-4 rounded-sm" style={{background: '#ff4d4f'}} /> above (depth &gt; 0°)</span>
              </>
            )}
            {isPoseAnalyzing && <span className="text-xs">analyzing...</span>}
            <button
              type="button"
              onClick={() => {
                const next: FacingDirection = facingDirection === 'right' ? 'left' : 'right';
                setFacingDirection(next);
                setKneeTrackingSide(facingDirectionToLeg(next));
              }}
              className="ml-auto flex items-center gap-0.5 text-[10px] opacity-40 hover:opacity-70 transition-opacity text-[var(--color-text-light)]"
              title={`Facing ${facingDirection} (auto-detected) — click to override`}
            >
              {facingDirection === 'right'
                ? <><span>Facing</span><ChevronRight className="h-3 w-3" /></>
                : <><ChevronLeft className="h-3 w-3" /><span>Facing</span></>
              }
            </button>
          </div>
        )}
      </section>
    );
  };

  /** Letterboxed media: tall single view; slightly shorter when two-up compare. */
  const mediaMaxClass = hasCompareMedia
    ? 'max-h-[min(52dvh,800px)] md:max-h-[min(58dvh,960px)] lg:max-h-[min(56dvh,1040px)]'
    : 'max-h-[min(82dvh,1200px)] md:max-h-[min(78dvh,1200px)] lg:max-h-[min(76dvh,1280px)]';

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
              className={`flex w-full min-w-0 flex-col gap-3 ${isFullscreen ? 'h-[100dvh] rounded-none p-2 md:p-3' : ''}`}
            >
              {/* Tall phone media: no fixed 16:9 box; cap desktop height so the page fits the window */}
              <div
                className={`flex w-full min-w-0 items-center justify-center overflow-hidden ${isFullscreen ? 'min-h-0 flex-1 rounded-none border-0 p-0' : 'min-h-[min(52dvh,560px)] rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-0'}`}
              >
                <div
                  className={`grid w-full min-w-0 items-stretch justify-items-stretch ${hasCompareMedia ? 'md:grid-cols-2 md:gap-0' : 'grid-cols-1 gap-0'}`}
                >
                  {/* Side-by-side: align portrait media to the inner seam (object-contain letterboxing otherwise sits in the middle). */}
                  <div
                    className={`flex min-w-0 flex-col gap-2 ${hasCompareMedia ? 'md:items-end' : 'items-center'}`}
                  >
                    <div
                      ref={mediaWrapRef}
                      className={`relative inline-block min-w-0 max-w-full overflow-hidden ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
                    >
                      <div
                        className={
                          appliedZoom
                            ? 'relative inline-block'
                            : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`
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
                          className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
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
                              : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`
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
                              className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
                            />
                          ) : compareImageSrc ? (
                            <img
                              ref={compareImageRef}
                              src={compareImageSrc}
                              className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
                              alt="Compare media"
                            />
                          ) : null}
                        </div>
                        <canvas
                          ref={compareCanvasRef}
                          className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                          style={{
                            touchAction: 'none',
                            cursor: isComparePanning ? 'grabbing' : activeTool === 'pan' ? 'grab' : 'default',
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
              </div>

              <div className="flex flex-col gap-2 rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-3 sm:p-4">
                <div className={`grid gap-2 ${compareVideoSrc ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
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
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setIsPlaying((v) => !v)}
                      className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
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
                      {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                    </button>
                    <button
                      type="button"
                      onClick={() => stepBothFrames(-1)}
                      className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                      title="Step both videos back one frame"
                      aria-label="Step both videos back one frame"
                    >
                      <ChevronLeft className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => stepBothFrames(1)}
                      className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                      title="Step both videos forward one frame"
                      aria-label="Step both videos forward one frame"
                    >
                      <ChevronRight className="w-6 h-6" />
                    </button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={handleDeleteMeasurements}
                      disabled={points.length === 0}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${points.length === 0 ? 'opacity-50 cursor-not-allowed' : 'text-red-400'}`}
                    >
                      <Trash className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={toggleRulerTool}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'ruler' ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
                      title="Angle ruler"
                      aria-label="Angle ruler"
                    >
                      <Ruler className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setPoseEnabled((v) => !v)}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${poseEnabled ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
                      title="Toggle pose overlay"
                      aria-label="Toggle pose overlay"
                    >
                      <Activity className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => zoomByButton(ZOOM_BUTTON_FACTOR)}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-fg`}
                      title="Zoom in"
                      aria-label="Zoom in"
                    >
                      <ZoomIn className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => zoomByButton(1 / ZOOM_BUTTON_FACTOR)}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-fg`}
                      title="Zoom out"
                      aria-label="Zoom out"
                    >
                      <ZoomOut className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={togglePanTool}
                      disabled={!isMediaLoaded || activeScale <= ZOOM_MIN_SCALE}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded || activeScale <= ZOOM_MIN_SCALE ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'pan' ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
                      title="Pan tool"
                      aria-label="Pan tool"
                    >
                      <Hand className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => void toggleFullscreen()}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : 'text-fg'}`}
                      title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                      aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                    >
                      {isFullscreen ? <Minimize className="w-6 h-6" /> : <Maximize className="w-6 h-6" />}
                    </button>
                  </div>
                </div>
              </div>
              {renderKneeAnglePanel()}
            </div>
          ) : imageSrc ? (
            <div
              ref={fullscreenTargetRef}
              className={`flex w-full min-w-0 flex-col gap-3 ${isFullscreen ? 'h-[100dvh] rounded-none p-2 md:p-3' : ''}`}
            >
              <div
                className={`flex w-full min-w-0 items-center justify-center overflow-hidden ${isFullscreen ? 'min-h-0 flex-1 rounded-none border-0 p-0' : 'min-h-[min(52dvh,560px)] rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-0'}`}
              >
                <div
                  className={`grid w-full min-w-0 items-stretch justify-items-stretch ${hasCompareMedia ? 'md:grid-cols-2 md:gap-0' : 'grid-cols-1 gap-0'}`}
                >
                  <div className={`min-w-0 ${hasCompareMedia ? 'md:flex md:flex-col md:items-end' : 'flex flex-col items-center'}`}>
                    <div
                      ref={mediaWrapRef}
                      className={`relative inline-block min-w-0 max-w-full overflow-hidden ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
                    >
                    <div
                      className={
                          compareAppliedZoom
                          ? 'relative inline-block'
                          : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`
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
                        className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
                        alt="Uploaded"
                        onLoad={() => setPoints([])}
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
                            : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`
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
                            className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
                          />
                        ) : compareImageSrc ? (
                          <img
                            ref={compareImageRef}
                            src={compareImageSrc}
                            className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[calc(100dvh-12rem)]' : mediaMaxClass}`}
                            alt="Compare media"
                          />
                        ) : null}
                      </div>
                      <canvas
                        ref={compareCanvasRef}
                        className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                        style={{
                          touchAction: 'none',
                          cursor: isComparePanning ? 'grabbing' : activeTool === 'pan' ? 'grab' : 'default',
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
              </div>

              <div className="flex flex-wrap items-center justify-end gap-2 rounded-xl border border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)] p-3 sm:p-4">
                <button
                  type="button"
                  onClick={handleDeleteMeasurements}
                  disabled={points.length === 0}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${points.length === 0 ? 'opacity-50 cursor-not-allowed' : 'text-red-400'}`}
                >
                  <Trash className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={toggleRulerTool}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'ruler' ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
                  title="Angle ruler"
                  aria-label="Angle ruler"
                >
                  <Ruler className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={() => setPoseEnabled((v) => !v)}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${poseEnabled ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
                  title="Toggle pose overlay"
                  aria-label="Toggle pose overlay"
                >
                  <Activity className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={() => zoomByButton(ZOOM_BUTTON_FACTOR)}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-fg`}
                  title="Zoom in"
                  aria-label="Zoom in"
                >
                  <ZoomIn className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={() => zoomByButton(1 / ZOOM_BUTTON_FACTOR)}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-fg`}
                  title="Zoom out"
                  aria-label="Zoom out"
                >
                  <ZoomOut className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={togglePanTool}
                  disabled={!isMediaLoaded || activeScale <= ZOOM_MIN_SCALE}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded || activeScale <= ZOOM_MIN_SCALE ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'pan' ? 'text-[var(--color-accent)] bg-[var(--color-panel-hover)]' : 'text-fg'}`}
                  title="Pan tool"
                  aria-label="Pan tool"
                >
                  <Hand className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={() => void toggleFullscreen()}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : 'text-fg'}`}
                  title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                  aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                  {isFullscreen ? <Minimize className="w-6 h-6" /> : <Maximize className="w-6 h-6" />}
                </button>
              </div>
              {renderKneeAnglePanel()}
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

        {angle !== null && (
          <section className="glass p-6 rounded-2xl border border-[var(--color-accent)]/10 shadow-lg accent-glow">
            <h2 className="mb-2 text-xl font-semibold text-fg brand-font">Analysis Result</h2>
            <p className="mb-1 text-sm text-[var(--color-text-light)]">Measured Angle</p>
            <p className="text-5xl font-bold text-[var(--color-accent)]">{angle.toFixed(1)}°</p>
          </section>
        )}
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
    </div>
  );
}


