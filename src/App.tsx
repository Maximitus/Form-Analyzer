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
  Ruler,
  Trash,
  ZoomIn,
  ZoomOut,
  Camera,
  Image as ImageGalleryIcon,
  Video,
} from 'lucide-react';

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

function normalizeZoomRect(ax: number, ay: number, bx: number, by: number) {
  const x = Math.min(ax, bx);
  const y = Math.min(ay, by);
  const w = Math.abs(bx - ax);
  const h = Math.abs(by - ay);
  return {x, y, w, h};
}

const ZOOM_MIN_SIZE = 24;
const ZOOM_HANDLE_RADIUS = 10;

function getZoomCornerIndex(
  px: number,
  py: number,
  r: {x: number; y: number; w: number; h: number},
): number {
  const corners: [number, number][] = [
    [r.x, r.y],
    [r.x + r.w, r.y],
    [r.x + r.w, r.y + r.h],
    [r.x, r.y + r.h],
  ];
  for (let i = 0; i < 4; i++) {
    const [cx, cy] = corners[i]!;
    if (Math.hypot(px - cx, py - cy) <= ZOOM_HANDLE_RADIUS + 4) {
      return i;
    }
  }
  return -1;
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

function adjustZoomRectByCorner(
  corner: number,
  px: number,
  py: number,
  r: {x: number; y: number; w: number; h: number},
) {
  const right = r.x + r.w;
  const bottom = r.y + r.h;
  if (corner === 0) {
    return normalizeZoomRect(
      Math.min(px, right - ZOOM_MIN_SIZE),
      Math.min(py, bottom - ZOOM_MIN_SIZE),
      right,
      bottom,
    );
  }
  if (corner === 1) {
    return normalizeZoomRect(
      r.x,
      Math.min(py, bottom - ZOOM_MIN_SIZE),
      Math.max(px, r.x + ZOOM_MIN_SIZE),
      bottom,
    );
  }
  if (corner === 2) {
    return normalizeZoomRect(
      r.x,
      r.y,
      Math.max(px, r.x + ZOOM_MIN_SIZE),
      Math.max(py, r.y + ZOOM_MIN_SIZE),
    );
  }
  return normalizeZoomRect(right, r.y, px, py);
}

export default function App() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaWrapRef = useRef<HTMLDivElement>(null);
  /** Gallery: images + videos from files — never use `capture` on this input. */
  const galleryPickInputRef = useRef<HTMLInputElement>(null);
  /** Native still-photo fallback only: `image/*` + `capture="environment"` via JS. */
  const imagePickInputRef = useRef<HTMLInputElement>(null);
  /** Native video-capture fallback only: `video/*` + `capture="environment"` via JS. */
  const videoPickInputRef = useRef<HTMLInputElement>(null);
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
  const [angle, setAngle] = useState<number | null>(null);
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);

  /** `ruler` = angle tool, `zoom` = marquee zoom */
  type AnalysisTool = 'ruler' | 'zoom' | null;
  const [activeTool, setActiveTool] = useState<AnalysisTool>(null);
  type ZoomPhase = 'idle' | 'drawing' | 'confirm';
  const [zoomPhase, setZoomPhase] = useState<ZoomPhase>('idle');
  const [zoomDraft, setZoomDraft] = useState<{x: number; y: number; w: number; h: number} | null>(
    null,
  );
  const [appliedZoom, setAppliedZoom] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
    vw: number;
    vh: number;
    s: number;
  } | null>(null);
  const zoomDrawStartRef = useRef<{x: number; y: number} | null>(null);
  const [draggingZoomCorner, setDraggingZoomCorner] = useState<number | null>(null);

  const resetZoomAll = () => {
    setAppliedZoom(null);
    setZoomPhase('idle');
    setZoomDraft(null);
    zoomDrawStartRef.current = null;
    setDraggingZoomCorner(null);
  };

  const cancelZoomMarquee = () => {
    setZoomPhase('idle');
    setZoomDraft(null);
    zoomDrawStartRef.current = null;
    setDraggingZoomCorner(null);
  };

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setPoints([]);
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

  const stepFrame = (direction: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime += direction * 0.033; // Approx 30fps
    }
  };

  useEffect(() => {
    if (videoRef.current) {
      if (isPlaying) videoRef.current.play();
      else videoRef.current.pause();
    }
  }, [isPlaying]);

  useEffect(() => {
    const wrap = mediaWrapRef.current;
    const target = videoRef.current ?? imageRef.current;
    if (!wrap || !target) return;

    const ro = new ResizeObserver(() => setMediaLayoutVersion((v) => v + 1));
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [videoSrc, imageSrc]);

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

    if (activeTool === 'zoom' && zoomDraft && (zoomPhase === 'drawing' || zoomPhase === 'confirm')) {
      const {x: zx, y: zy, w: zw, h: zh} = zoomDraft;
      const tl = contentToOverlay(zx, zy, zMap);
      const br = contentToOverlay(zx + zw, zy + zh, zMap);
      const ow = br.x - tl.x;
      const oh = br.y - tl.y;
      ctx.save();
      ctx.strokeStyle = '#ff8800';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(tl.x, tl.y, ow, oh);
      ctx.setLineDash([]);
      if (zoomPhase === 'confirm') {
        const corners: [number, number][] = [
          [zx, zy],
          [zx + zw, zy],
          [zx + zw, zy + zh],
          [zx, zy + zh],
        ];
        for (const [ccx, ccy] of corners) {
          const co = contentToOverlay(ccx, ccy, zMap);
          ctx.beginPath();
          ctx.arc(co.x, co.y, ZOOM_HANDLE_RADIUS, 0, Math.PI * 2);
          ctx.fillStyle = '#ff8800';
          ctx.fill();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }
      ctx.restore();
    }
  }, [points, videoSrc, imageSrc, angle, mediaLayoutVersion, activeTool, zoomPhase, zoomDraft, appliedZoom]);

  const [draggingPointIndex, setDraggingPointIndex] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

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

  const handleSeek = (e: ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      const v = videoRef.current;
      const cap =
        duration > 0 ? duration : getReliableVideoDuration(v) || v.duration || time;
      const clamped = Number.isFinite(cap) && cap > 0 ? Math.min(Math.max(0, time), cap) : time;
      v.currentTime = clamped;
      setCurrentTime(v.currentTime);
    }
  };

  const handleDeleteMeasurements = () => {
    setPoints([]);
    setAngle(null);
    setIsDrawing(false);
    setActiveTool(null);
  };

  const isMediaLoaded = !!videoSrc || !!imageSrc;

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

  const applyZoomConfirm = () => {
    if (!zoomDraft || !mediaWrapRef.current) return;
    const wrap = mediaWrapRef.current;
    const vw = wrap.clientWidth;
    const vh = wrap.clientHeight;
    const {x, y, w, h} = zoomDraft;
    if (w < ZOOM_MIN_SIZE || h < ZOOM_MIN_SIZE) return;
    const s = Math.min(vw / w, vh / h);
    setAppliedZoom({x, y, w, h, vw, vh, s});
    cancelZoomMarquee();
    setActiveTool(null);
  };

  const toggleRulerTool = () => {
    setActiveTool((prev) => {
      if (prev === 'ruler') {
        setIsDrawing(false);
        return null;
      }
      cancelZoomMarquee();
      setIsDrawing(true);
      return 'ruler';
    });
  };

  const toggleZoomTool = () => {
    setActiveTool((prev) => {
      if (prev === 'zoom') {
        cancelZoomMarquee();
        return null;
      }
      setIsDrawing(false);
      cancelZoomMarquee();
      return 'zoom';
    });
  };

  const handleCanvasPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const {ox, oy} = pointerToOverlayPx(e.clientX, e.clientY, canvas);
    const {x, y} = pointerToContent(e.clientX, e.clientY, canvas);

    if (activeTool === 'zoom' && !appliedZoom) {
      if (zoomPhase === 'confirm' && zoomDraft) {
        const ci = getZoomCornerIndex(x, y, zoomDraft);
        if (ci >= 0) {
          e.currentTarget.setPointerCapture(e.pointerId);
          setDraggingZoomCorner(ci);
          return;
        }
        return;
      }
      if (zoomPhase === 'idle') {
        zoomDrawStartRef.current = {x, y};
        setZoomPhase('drawing');
        setZoomDraft({x, y, w: 0, h: 0});
        e.currentTarget.setPointerCapture(e.pointerId);
        return;
      }
      if (zoomPhase === 'drawing') {
        e.currentTarget.setPointerCapture(e.pointerId);
        return;
      }
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
    const {x, y} = pointerToContent(e.clientX, e.clientY, canvas);

    if (draggingZoomCorner !== null && zoomDraft) {
      setZoomDraft(adjustZoomRectByCorner(draggingZoomCorner, x, y, zoomDraft));
      return;
    }

    if (zoomPhase === 'drawing' && zoomDrawStartRef.current) {
      const st = zoomDrawStartRef.current;
      setZoomDraft(normalizeZoomRect(st.x, st.y, x, y));
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
    const canvas = canvasRef.current;
    if (canvas && activeTool === 'zoom' && zoomPhase === 'drawing' && zoomDrawStartRef.current) {
      const {x, y} = pointerToContent(e.clientX, e.clientY, canvas);
      const st = zoomDrawStartRef.current;
      zoomDrawStartRef.current = null;
      const r = normalizeZoomRect(st.x, st.y, x, y);
      if (r.w >= ZOOM_MIN_SIZE && r.h >= ZOOM_MIN_SIZE) {
        setZoomDraft(r);
        setZoomPhase('confirm');
      } else {
        setZoomDraft(null);
        setZoomPhase('idle');
      }
    }

    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* ignore if not captured */
    }
    setDraggingZoomCorner(null);
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
              <div className="flex w-full min-h-[12rem] min-w-0 items-center justify-center overflow-hidden rounded-xl border border-[#ff8800]/10 bg-black p-2 sm:p-3">
                <div
                  ref={mediaWrapRef}
                  className={`relative overflow-hidden ${appliedZoom ? 'mx-auto shrink-0' : 'inline-block max-h-[min(82dvh,920px)] max-w-full md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                  style={
                    appliedZoom
                      ? {width: appliedZoom.vw, height: appliedZoom.vh}
                      : undefined
                  }
                >
                  <div
                    className={
                      appliedZoom
                        ? 'relative inline-block'
                        : 'relative inline-block max-h-[min(82dvh,920px)] max-w-full md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'
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
                      className="block h-auto max-h-[min(82dvh,920px)] w-full max-w-full object-contain md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]"
                    />
                  </div>
                  <canvas
                    ref={canvasRef}
                    className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                    style={{
                      touchAction: 'none',
                      cursor:
                        draggingZoomCorner !== null
                          ? 'grabbing'
                          : activeTool === 'zoom' && !appliedZoom
                            ? zoomPhase === 'confirm'
                              ? 'default'
                              : 'crosshair'
                            : 'crosshair',
                    }}
                    onPointerDown={handleCanvasPointerDown}
                    onPointerMove={handleCanvasPointerMove}
                    onPointerUp={handleCanvasPointerUp}
                    onPointerCancel={handleCanvasPointerUp}
                    onPointerLeave={handleCanvasPointerUp}
                  />
                </div>
              </div>

              {zoomPhase === 'confirm' && (
                <div className="flex flex-col gap-2 rounded-xl border border-[#ff8800]/10 bg-[var(--color-surface-deep)] p-3 sm:p-4">
                  <p className="text-sm text-[var(--color-text-light)]">
                    Drag any corner to adjust the zoom area, then confirm to enlarge for analysis.
                  </p>
                  <div className="flex flex-wrap justify-end gap-2">
                    <button
                      type="button"
                      onClick={cancelZoomMarquee}
                      className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-bg-dark)] px-4 py-2 text-sm hover:bg-[var(--color-panel-hover)]"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={applyZoomConfirm}
                      className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[var(--color-bg-dark)] hover:bg-[var(--color-accent-hover)]"
                    >
                      Confirm zoom
                    </button>
                  </div>
                </div>
              )}

              <div className="flex flex-col gap-2 rounded-xl border border-[#ff8800]/10 bg-[var(--color-surface-deep)] p-3 sm:p-4">
                <input
                  type="range"
                  min="0"
                  max={Math.max(duration, currentTime, 0.0001)}
                  value={Math.min(currentTime, Math.max(duration, currentTime, 0.0001))}
                  onChange={handleSeek}
                  className="w-full accent-[#ff8800]"
                />
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                  >
                    {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
                  </button>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => stepFrame(-1)}
                      className="p-2 rounded-full hover:bg-[var(--color-panel-hover)]"
                    >
                      <ChevronLeft />
                    </button>
                    <button
                      type="button"
                      onClick={() => stepFrame(1)}
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
                      onClick={toggleZoomTool}
                      disabled={!isMediaLoaded || !!appliedZoom}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded || appliedZoom ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'zoom' ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                      title={appliedZoom ? 'Reset zoom to use marquee again' : 'Zoom — drag a rectangle'}
                      aria-label="Zoom tool"
                    >
                      <ZoomIn className="w-6 h-6" />
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
              <div className="flex w-full min-h-[12rem] min-w-0 items-center justify-center overflow-hidden rounded-xl border border-[#ff8800]/10 bg-black p-2 sm:p-3">
                <div
                  ref={mediaWrapRef}
                  className={`relative overflow-hidden ${appliedZoom ? 'mx-auto shrink-0' : 'inline-block max-h-[min(82dvh,920px)] max-w-full md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'}`}
                  style={
                    appliedZoom
                      ? {width: appliedZoom.vw, height: appliedZoom.vh}
                      : undefined
                  }
                >
                  <div
                    className={
                      appliedZoom
                        ? 'relative inline-block'
                        : 'relative inline-block max-h-[min(82dvh,920px)] max-w-full md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]'
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
                      className="block h-auto max-h-[min(82dvh,920px)] w-full max-w-full object-contain md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]"
                      alt="Uploaded"
                      onLoad={() => setPoints([])}
                    />
                  </div>
                  <canvas
                    ref={canvasRef}
                    className="pointer-events-auto absolute inset-0 z-10 h-full w-full touch-none"
                    style={{
                      touchAction: 'none',
                      cursor:
                        draggingZoomCorner !== null
                          ? 'grabbing'
                          : activeTool === 'zoom' && !appliedZoom
                            ? zoomPhase === 'confirm'
                              ? 'default'
                              : 'crosshair'
                            : 'crosshair',
                    }}
                    onPointerDown={handleCanvasPointerDown}
                    onPointerMove={handleCanvasPointerMove}
                    onPointerUp={handleCanvasPointerUp}
                    onPointerCancel={handleCanvasPointerUp}
                    onPointerLeave={handleCanvasPointerUp}
                  />
                </div>
              </div>

              {zoomPhase === 'confirm' && (
                <div className="flex flex-col gap-2 rounded-xl border border-[#ff8800]/10 bg-[var(--color-surface-deep)] p-3 sm:p-4">
                  <p className="text-sm text-[var(--color-text-light)]">
                    Drag any corner to adjust the zoom area, then confirm to enlarge for analysis.
                  </p>
                  <div className="flex flex-wrap justify-end gap-2">
                    <button
                      type="button"
                      onClick={cancelZoomMarquee}
                      className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-bg-dark)] px-4 py-2 text-sm hover:bg-[var(--color-panel-hover)]"
                    >
                      Cancel
                    </button>
                    <button
                      type="button"
                      onClick={applyZoomConfirm}
                      className="rounded-lg border border-[#ff8800]/20 bg-[var(--color-accent)] px-4 py-2 text-sm font-medium text-[var(--color-bg-dark)] hover:bg-[var(--color-accent-hover)]"
                    >
                      Confirm zoom
                    </button>
                  </div>
                </div>
              )}

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
                  onClick={toggleZoomTool}
                  disabled={!isMediaLoaded || !!appliedZoom}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded || appliedZoom ? 'opacity-50 cursor-not-allowed' : ''} ${activeTool === 'zoom' ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                  title="Zoom — drag a rectangle"
                  aria-label="Zoom tool"
                >
                  <ZoomIn className="w-6 h-6" />
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
