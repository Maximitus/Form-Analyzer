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
import { Upload, Play, Pause, ChevronLeft, ChevronRight, Ruler, Trash, Camera } from 'lucide-react';
export default function App() {
  const [videoSrc, setVideoSrc] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mediaWrapRef = useRef<HTMLDivElement>(null);
  const [mediaLayoutVersion, setMediaLayoutVersion] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [angle, setAngle] = useState<number | null>(null);
  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);

  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setPoints([]);
      if (file.type.startsWith('video/')) {
        setVideoSrc(URL.createObjectURL(file));
        setImageSrc(null);
      } else if (file.type.startsWith('image/')) {
        setImageSrc(URL.createObjectURL(file));
        setVideoSrc(null);
      }
    }
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

    const media = video || image;
    canvas.width = media.clientWidth;
    canvas.height = media.clientHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#ff8800'; // Accent color
    ctx.fillStyle = '#ff8800';
    ctx.lineWidth = 3;

    if (points.length > 0) {
      points.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    if (points.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, points[0].y);
      ctx.lineTo(points[1].x, points[1].y);
      if (points.length === 3) {
        ctx.lineTo(points[2].x, points[2].y);
        
        // Draw angle text
        ctx.font = '20px Space Grotesk';
        ctx.fillStyle = '#ff8800';
        ctx.fillText(`${angle?.toFixed(1)}°`, points[1].x + 10, points[1].y - 10);
      }
      ctx.stroke();
    }
  }, [points, videoSrc, imageSrc, angle, mediaLayoutVersion]);

  const [draggingPointIndex, setDraggingPointIndex] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => setCurrentTime(video.currentTime);
    const handleLoadedMetadata = () => setDuration(video.duration);

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('loadedmetadata', handleLoadedMetadata);

    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [videoSrc]);

  const handleSeek = (e: ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const handleDeleteMeasurements = () => {
    setPoints([]);
    setAngle(null);
    setIsDrawing(false);
  };

  const isMediaLoaded = !!videoSrc || !!imageSrc;

  const getPointAt = (x: number, y: number) => {
    return points.findIndex(p => Math.hypot(p.x - x, p.y - y) < 15); // 15px hit radius
  };

  const handleCanvasPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const pointIndex = getPointAt(x, y);
    if (pointIndex !== -1) {
      e.currentTarget.setPointerCapture(e.pointerId);
      setDraggingPointIndex(pointIndex);
      return;
    }

    if (!isDrawing) return;

    e.currentTarget.setPointerCapture(e.pointerId);
    setPoints(prev => {
      if (prev.length >= 3) return [{ x, y }];
      const newPoints = [...prev, { x, y }];
      if (newPoints.length === 3) {
        calculateAngle(newPoints);
      }
      return newPoints;
    });
  };

  const handleCanvasPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (draggingPointIndex === null) return;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

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
            <div className="flex gap-2">
              <label className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20">
                <Upload className="w-6 h-6 text-[#ff8800]" />
                <input type="file" accept="video/mp4,video/webm,video/ogg,video/quicktime,image/jpeg,image/png,image/webp" className="hidden" onChange={handleFileUpload} />
              </label>
              <label className="cursor-pointer bg-[var(--color-bg-dark)] p-2 rounded-full hover:bg-[var(--color-panel-hover)] transition border border-[#ff8800]/20">
                <Camera className="w-6 h-6 text-[#ff8800]" />
                <input type="file" accept="video/*,image/*" capture="environment" className="hidden" onChange={handleFileUpload} />
              </label>
            </div>
          </div>

          {videoSrc ? (
            <div className="flex w-full min-w-0 flex-col gap-3">
              {/* Tall phone media: no fixed 16:9 box; cap desktop height so the page fits the window */}
              <div className="flex w-full min-h-[12rem] min-w-0 items-center justify-center overflow-hidden rounded-xl border border-[#ff8800]/10 bg-black p-2 sm:p-3">
                <div
                  ref={mediaWrapRef}
                  className="relative inline-block max-h-[min(82dvh,920px)] max-w-full md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]"
                >
                  <video
                    ref={videoRef}
                    src={videoSrc}
                    className="block h-auto max-h-[min(82dvh,920px)] w-full max-w-full object-contain md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]"
                  />
                  <canvas
                    ref={canvasRef}
                    className="pointer-events-auto absolute inset-0 h-full w-full cursor-crosshair touch-none"
                    style={{ touchAction: 'none' }}
                    onPointerDown={handleCanvasPointerDown}
                    onPointerMove={handleCanvasPointerMove}
                    onPointerUp={handleCanvasPointerUp}
                    onPointerCancel={handleCanvasPointerUp}
                    onPointerLeave={handleCanvasPointerUp}
                  />
                </div>
              </div>

              <div className="flex flex-col gap-2 rounded-xl border border-[#ff8800]/10 bg-[var(--color-surface-deep)] p-3 sm:p-4">
                <input
                  type="range"
                  min="0"
                  max={duration}
                  value={currentTime}
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
                  <div className="flex items-center gap-2">
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
                      onClick={() => setIsDrawing(!isDrawing)}
                      disabled={!isMediaLoaded}
                      className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${isDrawing ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                    >
                      <Ruler className="w-6 h-6" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : imageSrc ? (
            <div className="flex w-full min-w-0 flex-col gap-3">
              <div className="flex w-full min-h-[12rem] min-w-0 items-center justify-center overflow-hidden rounded-xl border border-[#ff8800]/10 bg-black p-2 sm:p-3">
                <div
                  ref={mediaWrapRef}
                  className="relative inline-block max-h-[min(82dvh,920px)] max-w-full md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]"
                >
                  <img
                    ref={imageRef}
                    src={imageSrc}
                    className="block h-auto max-h-[min(82dvh,920px)] w-full max-w-full object-contain md:max-h-[min(60vh,680px)] lg:max-h-[min(64vh,740px)]"
                    alt="Uploaded"
                    onLoad={() => setPoints([])}
                  />
                  <canvas
                    ref={canvasRef}
                    className="pointer-events-auto absolute inset-0 h-full w-full cursor-crosshair touch-none"
                    style={{ touchAction: 'none' }}
                    onPointerDown={handleCanvasPointerDown}
                    onPointerMove={handleCanvasPointerMove}
                    onPointerUp={handleCanvasPointerUp}
                    onPointerCancel={handleCanvasPointerUp}
                    onPointerLeave={handleCanvasPointerUp}
                  />
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
                  onClick={() => setIsDrawing(!isDrawing)}
                  disabled={!isMediaLoaded}
                  className={`p-2 rounded-full hover:bg-[var(--color-panel-hover)] ${!isMediaLoaded ? 'opacity-50 cursor-not-allowed' : ''} ${isDrawing ? 'text-[#ff8800] bg-[var(--color-panel-hover)]' : 'text-white'}`}
                >
                  <Ruler className="w-6 h-6" />
                </button>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 rounded-xl border-2 border-dashed border-[#ff8800]/20 bg-[var(--color-bg-dark)] p-8 text-center">
              <Upload className="w-12 h-12 text-[#ff8800]/50 mb-4" />
              <p className="text-[var(--color-text-light)]">Upload a video or image to start analysis</p>
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
    </div>
  );
}
