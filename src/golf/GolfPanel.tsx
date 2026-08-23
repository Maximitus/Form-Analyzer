import {formatAngle, formatSignedAngle} from './kinematics';
import {formatEventTime} from './report';
import {GOLF_CAMERA_LABEL} from './types';
import type {GolfCamera, GolfFrontalMetrics, GolfHandedness} from './types';
import type {GolfSwingReport} from './report';

interface GolfPanelProps {
  metrics: GolfFrontalMetrics | null;
  report: GolfSwingReport | null;
  camera: GolfCamera;
  handedness: GolfHandedness;
  currentTime?: number;
  onCameraChange: (camera: GolfCamera) => void;
  onHandednessChange: (handedness: GolfHandedness) => void;
  onJumpToTime?: (time: number) => void;
  isFullscreen?: boolean;
}

function Metric({label, value, title}: {label: string; value: string; title?: string}) {
  return (
    <p className="font-medium text-[var(--color-accent)]" title={title}>
      {label}: <span className="text-[var(--color-text-light)]">{value}</span>
    </p>
  );
}

export function GolfPanel({
  metrics,
  report,
  camera,
  handedness,
  currentTime,
  onCameraChange,
  onHandednessChange,
  onJumpToTime,
  isFullscreen = false,
}: GolfPanelProps) {
  const faceOn = camera === 'face-on';
  const box = `rounded-md border p-2.5 ${
    isFullscreen ? 'border-[var(--color-accent)]/15 bg-black/35' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/40'
  }`;
  const activeEventId = (() => {
    if (!report || report.events.length === 0 || currentTime == null) return null;
    const nearest = report.events.reduce((best, event) =>
      Math.abs(event.time - currentTime) < Math.abs(best.time - currentTime) ? event : best,
    );
    return Math.abs(nearest.time - currentTime) <= 0.12 ? nearest.id : null;
  })();

  return (
    <div className="mt-2 grid gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex overflow-hidden rounded-md border border-[var(--color-accent)]/20">
          <button
            type="button"
            onClick={() => onCameraChange('face-on')}
            className={`px-2 py-1 text-xs transition-colors ${
              camera === 'face-on'
                ? 'bg-[var(--color-accent)] font-semibold text-[var(--color-bg-dark)]'
                : 'text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)]'
            }`}
            title="Camera in front of the player — frontal / coronal plane"
          >
            Face-on
          </button>
          <button
            type="button"
            onClick={() => onCameraChange('down-the-line')}
            className={`px-2 py-1 text-xs transition-colors ${
              camera === 'down-the-line'
                ? 'bg-[var(--color-accent)] font-semibold text-[var(--color-bg-dark)]'
                : 'text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)]'
            }`}
            title="Camera behind the player along the target line — sagittal"
          >
            Down-the-line
          </button>
        </div>
        <div className="flex overflow-hidden rounded-md border border-[var(--color-accent)]/20">
          <button
            type="button"
            onClick={() => onHandednessChange('right')}
            className={`px-2 py-1 text-xs transition-colors ${
              handedness === 'right'
                ? 'bg-[var(--color-accent)] font-semibold text-[var(--color-bg-dark)]'
                : 'text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)]'
            }`}
            title="Right-handed: lead side is anatomical left"
          >
            Right-handed
          </button>
          <button
            type="button"
            onClick={() => onHandednessChange('left')}
            className={`px-2 py-1 text-xs transition-colors ${
              handedness === 'left'
                ? 'bg-[var(--color-accent)] font-semibold text-[var(--color-bg-dark)]'
                : 'text-[var(--color-accent)] hover:bg-[var(--color-panel-hover)]'
            }`}
            title="Left-handed: lead side is anatomical right"
          >
            Left-handed
          </button>
        </div>
      </div>

      {!isFullscreen ? (
        <p className="text-[11px] text-[var(--color-text-light)]">
          {GOLF_CAMERA_LABEL[camera]}. Press Scrub next to play to mark address, top, impact, and finish, then get one
          overall read — not per-frame tips.
        </p>
      ) : null}

      <div className={box}>
        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
          Swing key points
        </p>
        {!report || report.events.length === 0 ? (
          <p className="text-xs text-[var(--color-text-light)]">
            {report?.summary ?? 'Scrub a swing to identify address, top, impact, and follow-through.'}
          </p>
        ) : (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {report.events.map((event) => (
              <button
                key={event.id}
                type="button"
                onClick={() => onJumpToTime?.(event.time)}
                className={`rounded-md border p-2 text-left hover:border-[var(--color-accent)]/40 ${
                  activeEventId === event.id
                    ? 'border-[var(--color-accent)]/70 bg-[var(--color-accent)]/10'
                    : 'border-[var(--color-accent)]/15 bg-[var(--color-bg-dark)]/50'
                }`}
                title={`Jump to ${event.label}`}
              >
                <p className="text-[10px] font-semibold uppercase tracking-wide text-[var(--color-accent)]">
                  {event.label}
                </p>
                <p className="text-xs text-[var(--color-text-light)]">{formatEventTime(event.time)}</p>
                <p className="mt-1 text-[11px] text-[var(--color-text-light)] opacity-80">
                  {faceOn ? 'Shoulders' : 'Shoulder line'} {formatSignedAngle(event.metrics.shoulderTiltDeg)}
                  {' · '}
                  {faceOn ? 'pelvis' : 'pelvic line'} {formatSignedAngle(event.metrics.pelvicTiltDeg)}
                </p>
              </button>
            ))}
          </div>
        )}
      </div>

      <div className={box}>
        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
          Overall advice
        </p>
        {!report ? (
          <p className="text-xs text-[var(--color-text-light)]">Scrub the swing first. Advice is based on the key points, not every frame.</p>
        ) : (
          <div className="grid gap-2 text-xs text-[var(--color-text-light)]">
            <p className="opacity-80">{report.summary}</p>
            {report.advice.map((cue) => (
              <div key={cue.id}>
                <p
                  className={`font-medium ${
                    cue.level === 'flag'
                      ? 'text-red-300'
                      : cue.level === 'watch'
                        ? 'text-amber-200'
                        : 'text-[var(--color-accent)]'
                  }`}
                >
                  {cue.title}
                </p>
                <p className="opacity-80">{cue.detail}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {metrics && !isFullscreen ? (
        <div className={box}>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
            Current frame
          </p>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-[var(--color-text-light)] sm:grid-cols-4">
            <Metric label={faceOn ? 'Shoulder tilt' : 'Shoulder line'} value={formatSignedAngle(metrics.shoulderTiltDeg)} />
            <Metric label={faceOn ? 'Pelvic tilt' : 'Pelvic line'} value={formatSignedAngle(metrics.pelvicTiltDeg)} />
            <Metric label="Lead knee" value={formatAngle(metrics.leadKneeDeg)} />
            <Metric label="Trail knee" value={formatAngle(metrics.trailKneeDeg)} />
          </div>
        </div>
      ) : null}
    </div>
  );
}
