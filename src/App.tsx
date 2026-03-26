/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  useState,
  useRef,
  useEffect,
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
  Video,
  Link2,
  Unlink2,
} from 'lucide-react';
import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision';

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
const POSE_HIP_KNEE_ANKLE_IDS = [23, 24, 25, 26, 27, 28];
const POSE_LEG_CONNECTIONS: [number, number][] = [
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [23, 24],
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

type AppliedZoomMap = {x: number; y: number; s: number};

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
  const [compareIsPlaying, setCompareIsPlaying] = useState(false);
  const [isLinkedPlayback, setIsLinkedPlayback] = useState(true);
  const [angle, setAngle] = useState<number | null>(null);
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [poseEnabled, setPoseEnabled] = useState(false);
  const [poseKeypoints, setPoseKeypoints] = useState<{x: number; y: number; v: number}[]>([]);
  const [comparePoseKeypoints, setComparePoseKeypoints] = useState<{x: number; y: number; v: number}[]>([]);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const poseRafRef = useRef<number | null>(null);
  const [poseStatus, setPoseStatus] = useState<'idle' | 'loading' | 'ready' | 'error'>('idle');
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
  const [appliedZoom, setAppliedZoom] = useState<{x: number; y: number; s: number} | null>(null);
  const [isPanning, setIsPanning] = useState(false);
  const activePointersRef = useRef<Map<number, {clientX: number; clientY: number}>>(new Map());
  const panDragRef = useRef<{
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

  const resetZoomAll = () => {
    setAppliedZoom(null);
    setActiveTool((prev) => (prev === 'pan' ? null : prev));
    setIsPanning(false);
    panDragRef.current = null;
    pinchRef.current = null;
    activePointersRef.current.clear();
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
      setCompareIsPlaying(false);
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

  useEffect(() => {
    if (videoRef.current) {
      if (isPlaying) videoRef.current.play();
      else videoRef.current.pause();
    }
  }, [isPlaying]);

  useEffect(() => {
    if (isLinkedPlayback) {
      setCompareIsPlaying(isPlaying);
      return;
    }
    if (compareVideoRef.current) {
      if (compareIsPlaying) void compareVideoRef.current.play().catch(() => {});
      else compareVideoRef.current.pause();
    }
  }, [compareIsPlaying, isLinkedPlayback, isPlaying]);

  useEffect(() => {
    if (!isLinkedPlayback || !compareVideoRef.current || !videoRef.current) return;
    if (isPlaying) {
      void compareVideoRef.current.play().catch(() => {});
      return;
    }
    compareVideoRef.current.pause();
  }, [isPlaying, isLinkedPlayback, compareVideoSrc]);

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
      poseLandmarkerRef.current?.close();
      poseLandmarkerRef.current = null;
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

    const ensureLandmarker = async () => {
      if (poseLandmarkerRef.current) return poseLandmarkerRef.current;
      setPoseStatus('loading');
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
      );
      const landmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
      });
      poseLandmarkerRef.current = landmarker;
      setPoseStatus('ready');
      return landmarker;
    };

    const updatePoseFromResult = (
      result: any,
      setter: (next: {x: number; y: number; v: number}[]) => void,
    ) => {
      const landmarks = result?.landmarks?.[0];
      if (!landmarks || landmarks.length === 0) {
        setter([]);
        return;
      }
      const selected = POSE_HIP_KNEE_ANKLE_IDS.map((id) => {
        const p = landmarks[id];
        return {x: p.x, y: p.y, v: p.visibility ?? 1};
      });
      setter(selected);
    };

    const run = async () => {
      try {
        const landmarker = await ensureLandmarker();
        if (cancelled) return;

        const primaryVideo = videoRef.current;
        const compareVideo = compareVideoRef.current;
        const primaryImage = imageRef.current;
        const compareImage = compareImageRef.current;
        const hasAnyVideo = (!!videoSrc && !!primaryVideo) || (!!compareVideoSrc && !!compareVideo);

        // Handle images immediately (single detection), videos in a shared RAF loop.
        if (!hasAnyVideo) {
          if (imageSrc && primaryImage) {
            await landmarker.setOptions({runningMode: 'IMAGE'});
            const result = landmarker.detect(primaryImage);
            if (!cancelled) updatePoseFromResult(result, setPoseKeypoints);
          }
          if (compareImageSrc && compareImage) {
            await landmarker.setOptions({runningMode: 'IMAGE'});
            const result = landmarker.detect(compareImage);
            if (!cancelled) updatePoseFromResult(result, setComparePoseKeypoints);
          }
          return;
        }

        await landmarker.setOptions({runningMode: 'VIDEO'});

        if (imageSrc && primaryImage) {
          // Primary is an image: detect once and keep it.
          await landmarker.setOptions({runningMode: 'IMAGE'});
          const result = landmarker.detect(primaryImage);
          if (!cancelled) updatePoseFromResult(result, setPoseKeypoints);
          await landmarker.setOptions({runningMode: 'VIDEO'});
        }
        if (compareImageSrc && compareImage) {
          await landmarker.setOptions({runningMode: 'IMAGE'});
          const result = landmarker.detect(compareImage);
          if (!cancelled) updatePoseFromResult(result, setComparePoseKeypoints);
          await landmarker.setOptions({runningMode: 'VIDEO'});
        }

        const tick = () => {
          if (cancelled) return;

          if (videoSrc && primaryVideo) {
            const v = primaryVideo;
            if (v.readyState >= 2 && (v.currentTime !== lastPrimaryVideoTime || !v.paused)) {
              lastPrimaryVideoTime = v.currentTime;
              const result = landmarker.detectForVideo(v, performance.now());
              updatePoseFromResult(result, setPoseKeypoints);
            }
          }

          if (compareVideoSrc && compareVideo) {
            const v = compareVideo;
            if (v.readyState >= 2 && (v.currentTime !== lastCompareVideoTime || !v.paused)) {
              lastCompareVideoTime = v.currentTime;
              const result = landmarker.detectForVideo(v, performance.now());
              updatePoseFromResult(result, setComparePoseKeypoints);
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

    const vw = wrap.clientWidth;
    const vh = wrap.clientHeight;
    canvas.width = Math.max(1, Math.round(vw));
    canvas.height = Math.max(1, Math.round(vh));

    const zMap: AppliedZoomMap | null = appliedZoom
      ? {x: appliedZoom.x, y: appliedZoom.y, s: appliedZoom.s}
      : null;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#ff8800'; // Accent color
    ctx.fillStyle = '#ff8800';
    ctx.lineWidth = 3;

    if (points.length > 0) {
      points.forEach(p => {
        const o = contentToOverlay(p.x, p.y, zMap);
        ctx.beginPath();
        ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    if (poseEnabled && poseKeypoints.length === POSE_HIP_KNEE_ANKLE_IDS.length) {
      const byId = new Map<number, {x: number; y: number; v: number}>();
      POSE_HIP_KNEE_ANKLE_IDS.forEach((id, i) => {
        const p = poseKeypoints[i];
        if (p) byId.set(id, p);
      });

      ctx.save();
      ctx.strokeStyle = '#00d2ff';
      ctx.fillStyle = '#00d2ff';
      ctx.lineWidth = 2.5;
      for (const [aId, bId] of POSE_LEG_CONNECTIONS) {
        const a = byId.get(aId);
        const b = byId.get(bId);
        if (!a || !b || a.v < 0.25 || b.v < 0.25) continue;
        const aPx = mapPoseNormToCanvasOverlayPx(a.x, a.y, videoRef.current ?? imageRef.current, canvasRef.current);
        const bPx = mapPoseNormToCanvasOverlayPx(b.x, b.y, videoRef.current ?? imageRef.current, canvasRef.current);
        ctx.beginPath();
        ctx.moveTo(aPx.x, aPx.y);
        ctx.lineTo(bPx.x, bPx.y);
        ctx.stroke();
      }
      for (const p of poseKeypoints) {
        if (p.v < 0.25) continue;
        const o = mapPoseNormToCanvasOverlayPx(p.x, p.y, videoRef.current ?? imageRef.current, canvasRef.current);
        ctx.beginPath();
        ctx.arc(o.x, o.y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
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
        
        // Draw angle text
        ctx.font = '20px Space Grotesk';
        ctx.fillStyle = '#ff8800';
        ctx.fillText(`${angle?.toFixed(1)}°`, o1.x + 10, o1.y - 10);
      }
      ctx.stroke();
    }

  }, [points, videoSrc, imageSrc, angle, mediaLayoutVersion, appliedZoom, poseEnabled, poseKeypoints]);

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
    if (comparePoseKeypoints.length !== POSE_HIP_KNEE_ANKLE_IDS.length) return;

    const media = (video ?? image) as HTMLVideoElement | HTMLImageElement;

    const byId = new Map<number, {x: number; y: number; v: number}>();
    POSE_HIP_KNEE_ANKLE_IDS.forEach((id, i) => {
      const p = comparePoseKeypoints[i];
      if (p) byId.set(id, p);
    });

    ctx.save();
    ctx.strokeStyle = '#00d2ff';
    ctx.fillStyle = '#00d2ff';
    ctx.lineWidth = 2.5;

    for (const [aId, bId] of POSE_LEG_CONNECTIONS) {
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
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
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

  const toggleLinkedPlayback = () => {
    setIsLinkedPlayback((prev) => {
      const next = !prev;
      if (next) {
        const compare = compareVideoRef.current;
        if (compare) {
          setCompareIsPlaying(isPlaying);
          if (isPlaying) {
            void compare.play().catch(() => {});
          } else {
            compare.pause();
          }
        }
      }
      return next;
    });
  };

  const handleDeleteMeasurements = () => {
    setPoints([]);
    setAngle(null);
    setIsDrawing(false);
    setActiveTool(null);
  };

  const isMediaLoaded = !!videoSrc || !!imageSrc;
  const hasCompareMedia = !!compareVideoSrc || !!compareImageSrc;
  const currentScale = appliedZoom?.s ?? 1;

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

  const getContentSize = () => {
    const media = (videoRef.current ?? imageRef.current) as HTMLElement | null;
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

  const zoomAt = (factor: number, anchorContentX: number, anchorContentY: number) => {
    const size = getContentSize();
    if (!size) return;
    const prevS = appliedZoom?.s ?? ZOOM_MIN_SCALE;
    const prevX = appliedZoom?.x ?? 0;
    const prevY = appliedZoom?.y ?? 0;
    const nextS = clamp(prevS * factor, ZOOM_MIN_SCALE, ZOOM_MAX_SCALE);
    if (Math.abs(nextS - ZOOM_MIN_SCALE) < 0.001) {
      setAppliedZoom(null);
      setActiveTool((prev) => (prev === 'pan' ? null : prev));
      return;
    }
    const nextX = anchorContentX - ((anchorContentX - prevX) * prevS) / nextS;
    const nextY = anchorContentY - ((anchorContentY - prevY) * prevS) / nextS;
    setAppliedZoom(clampZoomViewport(nextX, nextY, nextS, size.w, size.h));
  };

  const zoomByButton = (factor: number) => {
    const size = getContentSize();
    if (!size) return;
    const centerOverlay = {
      ox: size.w / 2,
      oy: size.h / 2,
    };
    const anchor = overlayToContent(
      centerOverlay.ox,
      centerOverlay.oy,
      appliedZoom ? {x: appliedZoom.x, y: appliedZoom.y, s: appliedZoom.s} : null,
    );
    zoomAt(factor, anchor.x, anchor.y);
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
    const canvas = canvasRef.current;
    if (!canvas) return;
    const {ox, oy} = pointerToOverlayPx(e.clientX, e.clientY, canvas);
    const {x, y} = pointerToContent(e.clientX, e.clientY, canvas);

    activePointersRef.current.set(e.pointerId, {clientX: e.clientX, clientY: e.clientY});
    const pointers = Array.from(activePointersRef.current.values());
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
      const pointers = Array.from(activePointersRef.current.values());
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

  const calculateAngle = (pts: { x: number; y: number }[]) => {
    const [p1, p2, p3] = pts;
    const angle1 = Math.atan2(p1.y - p2.y, p1.x - p2.x);
    const angle2 = Math.atan2(p3.y - p2.y, p3.x - p2.x);
    let angle = Math.abs((angle1 - angle2) * (180 / Math.PI));
    if (angle > 180) angle = 360 - angle;
    setAngle(angle);
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg-dark)] text-white font-sans blueprint-bg">
      <header className="bg-[var(--color-chrome-bar)] px-4 py-4 md:px-8 mb-5 shadow-md border-b border-[var(--color-accent)]/20">
        <h1 className="text-2xl font-semibold tracking-tight text-[var(--color-accent)] brand-font leading-tight">
          Form Analyzer
        </h1>
      </header>

      <div className="grid gap-6 px-4 pt-3 pb-12 md:px-8 md:pt-5">
        <section className="glass p-6 rounded-2xl border border-[#ff8800]/10 shadow-lg orange-glow">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-white brand-font">Media Analysis</h2>
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={openGalleryPicker}
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20"
                title="Pick image or video from gallery"
                aria-label="Pick image or video from gallery"
              >
                <ImageGalleryIcon className="w-6 h-6 text-[#ff8800]" />
              </button>
              <button
                type="button"
                onClick={openComparePicker}
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20"
                title="Add compare media"
                aria-label="Add compare media"
              >
                <ImageGalleryIcon className="w-6 h-6 text-white" />
              </button>
              {hasCompareMedia ? (
                <button
                  type="button"
                  onClick={() => {
                    setCompareVideoSrc(null);
                    setCompareImageSrc(null);
                    setCompareIsPlaying(false);
                  }}
                  className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20 text-red-400"
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
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20"
                title="Take a photo"
                aria-label="Take a photo with the camera"
              >
                <Camera className="w-6 h-6 text-[#ff8800]" />
              </button>
              <button
                type="button"
                onClick={() => void openDeviceVideoRecord()}
                className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20"
                title="Record video with the camera"
                aria-label="Record video with the camera"
              >
                <Video className="w-6 h-6 text-[#ff8800]" />
              </button>
            </div>
          </div>

          {videoSrc ? (
            <div className="flex w-full min-w-0 flex-col gap-3">
              {/* Tall phone media: no fixed 16:9 box; cap desktop height so the page fits the window */}
              <div
                ref={fullscreenTargetRef}
                className={`flex w-full min-h-[12rem] min-w-0 items-center justify-center overflow-hidden bg-black ${isFullscreen ? 'h-[100dvh] rounded-none border-0 p-0' : 'rounded-xl border border-[#ff8800]/10 p-2 sm:p-3'}`}
              >
                <div className={`grid w-full items-center justify-items-center gap-3 ${hasCompareMedia ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                  <div className="flex min-w-0 flex-col gap-2">
                    <div
                      ref={mediaWrapRef}
                      className={`relative inline-block max-w-full overflow-hidden ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                    >
                      <div
                        className={
                          appliedZoom
                            ? 'relative inline-block'
                            : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`
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
                          className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
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
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                        title="Play or pause primary video"
                        aria-label="Play or pause primary video"
                      >
                        {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                      </button>
                      <input
                        type="range"
                        min="0"
                        max={Math.max(duration, currentTime, 0.0001)}
                        value={Math.min(currentTime, Math.max(duration, currentTime, 0.0001))}
                        onChange={handleSeek}
                        step={SLIDER_SCRUB_STEP_SECONDS}
                        className="w-full accent-[#ff8800]"
                      />
                    </div>
                    <div className="flex items-center justify-center gap-2">
                      <button
                        type="button"
                        onClick={() => stepPrimaryFrame(-1)}
                        disabled={!videoSrc}
                        className="p-1.5 rounded-full hover:bg-[var(--color-panel-hover)]"
                        title="Step back one frame"
                        aria-label="Step back one frame"
                      >
                        <ChevronLeft className="w-5 h-5" />
                      </button>
                      <button
                        type="button"
                        onClick={() => stepPrimaryFrame(1)}
                        disabled={!videoSrc}
                        className="p-1.5 rounded-full hover:bg-[var(--color-panel-hover)]"
                        title="Step forward one frame"
                        aria-label="Step forward one frame"
                      >
                        <ChevronRight className="w-5 h-5" />
                      </button>
                    </div>
                  </div>
                  {hasCompareMedia ? (
                    <div className="flex min-w-0 flex-col gap-2">
                      <div
                        ref={compareMediaWrapRef}
                        className="relative inline-block max-w-full overflow-hidden"
                      >
                        {compareVideoSrc ? (
                          <video
                            ref={compareVideoRef}
                            src={compareVideoSrc}
                            className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                          />
                        ) : compareImageSrc ? (
                          <img
                            ref={compareImageRef}
                            src={compareImageSrc}
                            className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                            alt="Compare media"
                          />
                        ) : null}
                        <canvas
                          ref={compareCanvasRef}
                          className="pointer-events-none absolute inset-0 z-10 h-full w-full touch-none"
                          style={{touchAction: 'none'}}
                        />
                      </div>
                      {compareVideoSrc ? (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center gap-2">
                            {!isLinkedPlayback ? (
                              <button
                                type="button"
                                onClick={() => setCompareIsPlaying((v) => !v)}
                                className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                                title="Play or pause compare video"
                                aria-label="Play or pause compare video"
                              >
                                {compareIsPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                              </button>
                            ) : null}
                            <input
                              type="range"
                              min="0"
                              max={Math.max(compareDuration, compareCurrentTime, 0.0001)}
                              value={Math.min(compareCurrentTime, Math.max(compareDuration, compareCurrentTime, 0.0001))}
                              onChange={handleCompareSeek}
                              step={SLIDER_SCRUB_STEP_SECONDS}
                              className="w-full accent-[#ff8800]"
                            />
                          </div>
                          <div className="flex items-center justify-center gap-2">
                            <button
                              type="button"
                              onClick={() => stepCompareFrame(-1)}
                              className="p-1.5 rounded-full hover:bg-[var(--color-panel-hover)]"
                              title="Step back one frame"
                              aria-label="Step back one frame"
                            >
                              <ChevronLeft className="w-5 h-5" />
                            </button>
                            <button
                              type="button"
                              onClick={() => stepCompareFrame(1)}
                              className="p-1.5 rounded-full hover:bg-[var(--color-panel-hover)]"
                              title="Step forward one frame"
                              aria-label="Step forward one frame"
                            >
                              <ChevronRight className="w-5 h-5" />
                            </button>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="flex flex-col gap-2 rounded-xl border border-[#ff8800]/10 bg-[var(--color-surface-deep)] p-3 sm:p-4">
                {compareVideoSrc ? (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-[var(--color-text-light)]">Compare video timeline</span>
                      <button
                        type="button"
                        onClick={toggleLinkedPlayback}
                        className={`flex items-center gap-1 rounded-md px-2 py-1 text-xs ${isLinkedPlayback ? 'bg-[#ff8800] text-[var(--color-bg-dark)]' : 'bg-[var(--color-bg-dark)] text-white border border-[#ff8800]/20'}`}
                        title={isLinkedPlayback ? 'Unlink playback' : 'Link playback'}
                        aria-label={isLinkedPlayback ? 'Unlink playback' : 'Link playback'}
                      >
                        {isLinkedPlayback ? <Link2 className="h-3.5 w-3.5" /> : <Unlink2 className="h-3.5 w-3.5" />}
                        {isLinkedPlayback ? 'Linked' : 'Unlinked'}
                      </button>
                    </div>
                  </>
                ) : null}
                {poseEnabled ? (
                  <p className="text-xs text-[var(--color-text-light)]">
                    {poseStatus === 'loading'
                      ? 'Loading pose model...'
                      : poseStatus === 'error'
                        ? 'Pose model failed to load.'
                        : 'Pose map active: hips, knees, ankles'}
                  </p>
                ) : null}
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => stepBothFrames(-1)}
                      className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                    >
                      <ChevronLeft />
                    </button>
                    <button
                      type="button"
                      onClick={() => stepBothFrames(1)}
                      className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                    >
                      <ChevronRight />
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
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'ruler' ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                      title="Angle ruler"
                      aria-label="Angle ruler"
                    >
                      <Ruler className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setPoseEnabled((v) => !v)}
                      disabled={!isMediaLoaded}
                      className={`rounded-lg border border-[#ff8800]/20 px-3 py-1.5 text-sm hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${poseEnabled ? 'bg-[var(--color-panel-hover)] text-[#ff8800]' : 'text-white'}`}
                      title="Toggle BlazePose joints"
                      aria-label="Toggle BlazePose joints"
                    >
                      Pose
                    </button>
                    <button
                      type="button"
                      onClick={() => zoomByButton(ZOOM_BUTTON_FACTOR)}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-white`}
                      title="Zoom in"
                      aria-label="Zoom in"
                    >
                      <ZoomIn className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => zoomByButton(1 / ZOOM_BUTTON_FACTOR)}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-white`}
                      title="Zoom out"
                      aria-label="Zoom out"
                    >
                      <ZoomOut className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={togglePanTool}
                      disabled={!isMediaLoaded || currentScale <= ZOOM_MIN_SCALE}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded || currentScale <= ZOOM_MIN_SCALE ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'pan' ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                      title="Pan tool"
                      aria-label="Pan tool"
                    >
                      <Hand className="w-6 h-6" />
                    </button>
                    <button
                      type="button"
                      onClick={() => void toggleFullscreen()}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : 'text-white'}`}
                      title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                      aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                    >
                      {isFullscreen ? <Minimize className="w-6 h-6" /> : <Maximize className="w-6 h-6" />}
                    </button>
                    {appliedZoom ? (
                      <button
                        type="button"
                        onClick={resetZoomAll}
                        className="p-2 rounded-full hover:bg-[var(--color-panel-hover)] text-[#ff8800]"
                        title="Reset zoom"
                        aria-label="Reset zoom"
                      >
                        <ZoomOut className="w-6 h-6" />
                      </button>
                    ) : null}
                  </div>
                </div>
              </div>
            </div>
          ) : imageSrc ? (
            <div className="flex w-full min-w-0 flex-col gap-3">
              <div
                ref={fullscreenTargetRef}
                className={`flex w-full min-h-[12rem] min-w-0 items-center justify-center overflow-hidden bg-black ${isFullscreen ? 'h-[100dvh] rounded-none border-0 p-0' : 'rounded-xl border border-[#ff8800]/10 p-2 sm:p-3'}`}
              >
                <div className={`grid w-full items-center justify-items-center gap-3 ${hasCompareMedia ? 'md:grid-cols-2' : 'grid-cols-1'}`}>
                  <div
                    ref={mediaWrapRef}
                    className={`relative inline-block max-w-full overflow-hidden ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                  >
                    <div
                      className={
                        appliedZoom
                          ? 'relative inline-block'
                          : `relative inline-block max-w-full ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`
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
                      <img
                        ref={imageRef}
                        src={imageSrc}
                        className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
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
                  {hasCompareMedia ? (
                    <div
                      ref={compareMediaWrapRef}
                      className="relative inline-block max-w-full overflow-hidden"
                    >
                      {compareVideoSrc ? (
                        <video
                          ref={compareVideoRef}
                          src={compareVideoSrc}
                          className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                        />
                      ) : compareImageSrc ? (
                        <img
                          ref={compareImageRef}
                          src={compareImageSrc}
                          className={`block h-auto w-full max-w-full object-contain ${isFullscreen ? 'max-h-[100dvh]' : 'max-h-[min(82dvh,920px)] md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                          alt="Compare media"
                        />
                      ) : null}
                      <canvas
                        ref={compareCanvasRef}
                        className="pointer-events-none absolute inset-0 z-10 h-full w-full touch-none"
                        style={{touchAction: 'none'}}
                      />
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="flex flex-wrap items-center justify-end gap-2 rounded-xl border border-[#ff8800]/10 bg-[var(--color-surface-deep)] p-3 sm:p-4">
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
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'ruler' ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                  title="Angle ruler"
                  aria-label="Angle ruler"
                >
                  <Ruler className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={() => setPoseEnabled((v) => !v)}
                  disabled={!isMediaLoaded}
                  className={`rounded-lg border border-[#ff8800]/20 px-3 py-1.5 text-sm hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${poseEnabled ? 'bg-[var(--color-panel-hover)] text-[#ff8800]' : 'text-white'}`}
                  title="Toggle BlazePose joints"
                  aria-label="Toggle BlazePose joints"
                >
                  Pose
                </button>
                <button
                  type="button"
                  onClick={() => zoomByButton(ZOOM_BUTTON_FACTOR)}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-white`}
                  title="Zoom in"
                  aria-label="Zoom in"
                >
                  <ZoomIn className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={() => zoomByButton(1 / ZOOM_BUTTON_FACTOR)}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} text-white`}
                  title="Zoom out"
                  aria-label="Zoom out"
                >
                  <ZoomOut className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={togglePanTool}
                  disabled={!isMediaLoaded || currentScale <= ZOOM_MIN_SCALE}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded || currentScale <= ZOOM_MIN_SCALE ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'pan' ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                  title="Pan tool"
                  aria-label="Pan tool"
                >
                  <Hand className="w-6 h-6" />
                </button>
                <button
                  type="button"
                  onClick={() => void toggleFullscreen()}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : 'text-white'}`}
                  title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                  aria-label={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
                >
                  {isFullscreen ? <Minimize className="w-6 h-6" /> : <Maximize className="w-6 h-6" />}
                </button>
                {appliedZoom ? (
                  <button
                    type="button"
                    onClick={resetZoomAll}
                    className="p-2 rounded-full hover:bg-[var(--color-panel-hover)] text-[#ff8800]"
                    title="Reset zoom"
                    aria-label="Reset zoom"
                  >
                    <ZoomOut className="w-6 h-6" />
                  </button>
                ) : null}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 rounded-xl border-2 border-dashed border-[#ff8800]/20 bg-[var(--color-bg-dark)] p-8 text-center">
              <div className="mb-4 flex gap-3 text-[#ff8800]/50">
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
          <section className="glass p-6 rounded-2xl border border-[#ff8800]/10 shadow-lg orange-glow">
            <h2 className="text-xl font-semibold text-white brand-font mb-2">Analysis Result</h2>
            <p className="text-sm text-white/70 mb-1">Measured Angle</p>
            <p className="text-5xl font-bold text-[#ff8800]">{angle.toFixed(1)}°</p>
          </section>
        )}
      </div>

      {cameraModalOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="camera-dialog-title"
        >
          <div className="glass orange-glow w-full max-w-lg rounded-2xl border border-[#ff8800]/10 p-4 shadow-xl">
            <h3 id="camera-dialog-title" className="mb-3 text-lg font-semibold text-white brand-font">
              Camera
            </h3>
            <div className="overflow-hidden rounded-xl border border-[#ff8800]/10 bg-black">
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
                className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-bg-dark)] px-4 py-2 text-sm hover:bg-[var(--color-panel-hover)]"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={capturePhotoFromCamera}
                className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[var(--color-bg-dark)] hover:bg-[var(--color-accent-hover)]"
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
          <div className="glass orange-glow w-full max-w-lg rounded-2xl border border-[#ff8800]/10 p-4 shadow-xl">
            <h3 id="video-record-dialog-title" className="mb-3 text-lg font-semibold text-white brand-font">
              Record video
            </h3>
            <div className="overflow-hidden rounded-xl border border-[#ff8800]/10 bg-black">
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
                className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-bg-dark)] px-4 py-2 text-sm hover:bg-[var(--color-panel-hover)]"
              >
                Cancel
              </button>
              {!isRecordingVideo ? (
                <button
                  type="button"
                  onClick={startVideoRecording}
                  className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[var(--color-bg-dark)] hover:bg-[var(--color-accent-hover)]"
                >
                  Start recording
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopVideoRecordingAndSave}
                  className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[var(--color-bg-dark)] hover:bg-[var(--color-accent-hover)]"
                >
                  Stop and use clip
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
