/** COCO 17-keypoint names, matching RTMPose SimCC body models. */
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

export type CocoKeypointName = (typeof COCO_KEYPOINT_NAMES)[number];

export const COCO_KEYPOINT_COUNT = COCO_KEYPOINT_NAMES.length;

export const COCO_INDEX = {
  nose: 0,
  leftEye: 1,
  rightEye: 2,
  leftEar: 3,
  rightEar: 4,
  leftShoulder: 5,
  rightShoulder: 6,
  leftElbow: 7,
  rightElbow: 8,
  leftWrist: 9,
  rightWrist: 10,
  leftHip: 11,
  rightHip: 12,
  leftKnee: 13,
  rightKnee: 14,
  leftAnkle: 15,
  rightAnkle: 16,
} as const;

/** Skeleton edges as pairs of COCO indices. */
export const COCO_SKELETON: ReadonlyArray<readonly [number, number]> = [
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [0, 5],
  [0, 6],
];

export interface Point2D {
  x: number;
  y: number;
}

export interface Keypoint2D extends Point2D {
  score: number;
  name: CocoKeypointName;
}

export interface ClubHeadEstimate {
  x: number;
  y: number;
  score: number;
  gripX: number;
  gripY: number;
  method: 'image' | 'prior';
}

export interface PoseFrame {
  frameId: number;
  mediaTime: number;
  inferenceMs: number;
  width: number;
  height: number;
  keypoints: Keypoint2D[];
  clubHead?: ClubHeadEstimate;
}

export interface ClinicalAngles {
  leftHip: number | null;
  rightHip: number | null;
  leftKnee: number | null;
  rightKnee: number | null;
  leftAnkle: number | null;
  rightAnkle: number | null;
  trunk: number | null;
}

export type RtmposeVariant = 's' | 'm';

export interface LetterboxMeta {
  scale: number;
  padX: number;
  padY: number;
  srcWidth: number;
  srcHeight: number;
  dstWidth: number;
  dstHeight: number;
}

export interface ModelSpec {
  variant: RtmposeVariant;
  /** Local public path, preferred when the file has been downloaded. */
  localUrl: string;
  /** Remote ONNX (SimCC, 17 COCO keypoints, 256x192). */
  remoteUrl: string;
  inputWidth: number;
  inputHeight: number;
  simccSplitRatio: number;
}

export const RTMPOSE_MODELS: Record<RtmposeVariant, ModelSpec> = {
  s: {
    variant: 's',
    localUrl: '/models/rtmpose-s.onnx',
    remoteUrl:
      'https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-s.onnx',
    inputWidth: 192,
    inputHeight: 256,
    simccSplitRatio: 2,
  },
  m: {
    variant: 'm',
    localUrl: '/models/rtmpose-m.onnx',
    remoteUrl:
      'https://huggingface.co/bukuroo/RTMPose-ONNX/resolve/main/rtmpose-m.onnx',
    inputWidth: 192,
    inputHeight: 256,
    simccSplitRatio: 2,
  },
};

export const IMAGENET_MEAN = [123.675, 116.28, 103.53] as const;
export const IMAGENET_STD = [58.395, 57.12, 57.375] as const;
