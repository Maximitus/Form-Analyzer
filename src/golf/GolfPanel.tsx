import {formatAngle, formatSignedAngle} from './kinematics';
import {GOLF_CAMERA_LABEL, GOLF_PHASE_LABEL} from './types';
import type {GolfCamera, GolfFrontalMetrics, GolfHandedness} from './types';

interface GolfPanelProps {
  metrics: GolfFrontalMetrics | null;
  camera: GolfCamera;
  handedness: GolfHandedness;
  onCameraChange: (camera: GolfCamera) => void;
  onHandednessChange: (handedness: GolfHandedness) => void;
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
  camera,
  handedness,
  onCameraChange,
  onHandednessChange,
  isFullscreen = false,
}: GolfPanelProps) {
  const faceOn = camera === 'face-on';
  const box = `rounded-md border p-2.5 ${
    isFullscreen ? 'border-transparent bg-black/50 backdrop-blur-md' : 'border-[var(--color-accent)]/10 bg-[var(--color-bg-dark)]/40'
  }`;

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

      <p className="text-[11px] text-[var(--color-text-light)]">
        {GOLF_CAMERA_LABEL[camera]}. Play or scrub to sample pose. Club head is a wrist-guided search —
        the pose model only outputs body landmarks.
      </p>

      <div className={box}>
        {!metrics ? (
          <p className="text-xs text-[var(--color-text-light)]">Enable pose and play the video to collect golf samples.</p>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs text-[var(--color-text-light)] sm:grid-cols-4">
              <Metric
                label="Phase"
                value={`${GOLF_PHASE_LABEL[metrics.phase]}${metrics.calibrated ? '' : ' · calibrating'}`}
              />
              <Metric label="Lead" value={`${metrics.leadSide} (${metrics.handedness}-handed)`} />
              <Metric
                label={faceOn ? 'Shoulder tilt' : 'Shoulder line'}
                value={formatSignedAngle(metrics.shoulderTiltDeg)}
                title="Anatomical left high is positive"
              />
              <Metric
                label={faceOn ? 'Pelvic tilt' : 'Pelvic line'}
                value={formatSignedAngle(metrics.pelvicTiltDeg)}
              />
              <Metric
                label={faceOn ? 'Side-bend' : 'Spine vs vertical'}
                value={formatSignedAngle(metrics.lateralTrunkFlexionDeg)}
              />
              <Metric
                label={faceOn ? 'Head sway' : 'Head vs ball line'}
                value={
                  metrics.headSwayTowardLeadPct === null
                    ? '—'
                    : `${metrics.headSwayTowardLeadPct.toFixed(0)}% toward lead`
                }
              />
              <Metric
                label={faceOn ? 'Hip sway' : 'Hip vs ball line'}
                value={
                  metrics.hipSwayTowardLeadPct === null
                    ? '—'
                    : `${metrics.hipSwayTowardLeadPct.toFixed(0)}% toward lead`
                }
              />
              <Metric label="Lead knee" value={formatAngle(metrics.leadKneeDeg)} />
              <Metric label="Trail knee" value={formatAngle(metrics.trailKneeDeg)} />
              <Metric label="Lead elbow" value={formatAngle(metrics.leadElbowDeg)} />
              <Metric label="Trail elbow" value={formatAngle(metrics.trailElbowDeg)} />
              <Metric
                label="Shaft"
                value={formatSignedAngle(metrics.shaftFromVerticalDeg)}
                title="Club shaft vs downward vertical"
              />
              <Metric
                label="Club"
                value={
                  !metrics.clubHead
                    ? '—'
                    : metrics.clubHead.method === 'image'
                      ? 'tracked'
                      : 'shaft prior'
                }
              />
              <Metric
                label="Club vs midline"
                value={
                  metrics.clubHeadTowardLeadPct === null
                    ? '—'
                    : `${metrics.clubHeadTowardLeadPct.toFixed(0)}% toward lead`
                }
              />
              <Metric
                label="Club speed"
                value={
                  metrics.clubHeadSpeedPxPerSec === null ? '—' : `${metrics.clubHeadSpeedPxPerSec.toFixed(0)} px/s`
                }
              />
              <Metric
                label="Stance / shoulders"
                value={metrics.stanceToShoulderRatio === null ? '—' : `${metrics.stanceToShoulderRatio.toFixed(2)}×`}
              />
            </div>
          </>
        )}
      </div>

      {metrics && metrics.cues.length > 0 ? (
        <div className={box}>
          <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--color-text-light)] opacity-70">
            Cues
          </p>
          <div className="grid gap-2 text-xs text-[var(--color-text-light)] md:grid-cols-2">
            {metrics.cues.map((cue) => (
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
        </div>
      ) : null}
    </div>
  );
}
