import {trackedPoseToCoco} from './fromTrackedPose';
import {computeGolfFrontalMetrics} from './metrics';
import type {
  GolfCamera,
  GolfCue,
  GolfFrontalMetrics,
  GolfHandedness,
  GolfPhase,
} from './types';
import {GOLF_PHASE_LABEL} from './types';

export type GolfSwingEventId = Extract<GolfPhase, 'address' | 'top' | 'impact' | 'follow-through'>;

export interface GolfSwingEvent {
  id: GolfSwingEventId;
  label: string;
  time: number;
  metrics: GolfFrontalMetrics;
}

export interface GolfSwingReport {
  events: GolfSwingEvent[];
  advice: GolfCue[];
  summary: string;
}

interface Sample {
  time: number;
  metrics: GolfFrontalMetrics;
}

const MIN_VIS = 0.15;
const SCALE = 1000;

function abs(n: number | null): number | null {
  return n === null ? null : Math.abs(n);
}

function argMin(samples: Sample[], start: number, end: number, value: (s: Sample) => number | null): number {
  let best = -1;
  let bestVal = Infinity;
  for (let i = start; i <= end; i++) {
    const v = value(samples[i]!);
    if (v === null) continue;
    if (v < bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

function argMax(samples: Sample[], start: number, end: number, value: (s: Sample) => number | null): number {
  let best = -1;
  let bestVal = -Infinity;
  for (let i = start; i <= end; i++) {
    const v = value(samples[i]!);
    if (v === null) continue;
    if (v > bestVal) {
      bestVal = v;
      best = i;
    }
  }
  return best;
}

function elev(s: Sample): number | null {
  return s.metrics.wristElevation;
}

/**
 * Detect address, top, impact, and follow-through from a scrubbed pose cache.
 * Uses wrist height (relative to torso) as the primary swing clock.
 */
export function buildGolfSwingReport(
  frames: readonly {time: number; keypoints: readonly {x: number; y: number; v: number}[]}[],
  trackedIds: readonly number[],
  handedness: GolfHandedness,
  camera: GolfCamera,
): GolfSwingReport {
  const samples: Sample[] = [];
  for (const frame of frames) {
    if (!frame.keypoints.some((p) => p.v >= MIN_VIS)) continue;
    const coco = trackedPoseToCoco(frame.keypoints, trackedIds, SCALE, SCALE);
    const metrics = computeGolfFrontalMetrics(coco, handedness, 'unknown', null, {camera});
    if (metrics.wristElevation === null) continue;
    samples.push({time: frame.time, metrics});
  }

  if (samples.length < 8) {
    return {
      events: [],
      advice: [
        {
          id: 'need-swing',
          level: 'info',
          title: 'Need a full swing',
          detail:
            'Scrub a clip that shows setup through the finish. Key points are address, top, impact, and follow-through.',
        },
      ],
      summary: 'Not enough pose samples to mark swing key points. Scrub a complete swing, then review the report.',
    };
  }

  const last = samples.length - 1;
  const earlyEnd = Math.max(1, Math.floor(samples.length * 0.35));
  const addressIdx = argMin(samples, 0, earlyEnd, (s) => {
    const e = elev(s);
    return e !== null && e < 0.55 ? e : null;
  });
  const afterAddress = Math.max(0, addressIdx);
  const topIdx = argMax(samples, afterAddress, last, elev);
  const peak = topIdx >= 0 ? elev(samples[topIdx]!) : null;
  const afterTop = topIdx >= 0 ? topIdx : afterAddress;
  const impactIdx = argMin(samples, afterTop, last, (s) => {
    const e = elev(s);
    if (e === null || peak === null) return e;
    return e <= peak * 0.72 ? e : null;
  });
  const afterImpact = impactIdx >= 0 ? impactIdx : afterTop;
  const finishIdx = argMax(samples, afterImpact, last, elev);

  const events: GolfSwingEvent[] = [];
  const pushEvent = (id: GolfSwingEventId, idx: number) => {
    const sample = samples[idx];
    if (!sample) return;
    if (events.some((e) => Math.abs(e.time - sample.time) < 0.04)) return;
    events.push({
      id,
      label: GOLF_PHASE_LABEL[id],
      time: sample.time,
      metrics: {...sample.metrics, phase: id, cues: []},
    });
  };

  if (addressIdx >= 0) pushEvent('address', addressIdx);
  if (topIdx >= 0 && topIdx !== addressIdx) pushEvent('top', topIdx);
  if (impactIdx >= 0 && impactIdx !== topIdx) pushEvent('impact', impactIdx);
  if (finishIdx >= 0 && finishIdx !== impactIdx && finishIdx !== topIdx) {
    pushEvent('follow-through', finishIdx);
  }

  const advice = buildOverallAdvice(events, camera);
  const found = events.map((e) => e.label).join(', ');
  const summary =
    events.length >= 3
      ? `Marked ${found}. Advice below is from those key points, not every frame.`
      : events.length > 0
        ? `Only found ${found}. Scrub a clearer full-swing clip to mark the rest.`
        : 'Could not mark address, top, impact, or finish from this clip.';

  return {events, advice, summary};
}

function byId(events: GolfSwingEvent[], id: GolfSwingEventId): GolfSwingEvent | undefined {
  return events.find((e) => e.id === id);
}

function buildOverallAdvice(events: GolfSwingEvent[], camera: GolfCamera): GolfCue[] {
  const advice: GolfCue[] = [];
  const faceOn = camera === 'face-on';
  const address = byId(events, 'address');
  const top = byId(events, 'top');
  const impact = byId(events, 'impact');
  const finish = byId(events, 'follow-through');

  if (!address && !top) {
    advice.push({
      id: 'no-keys',
      level: 'info',
      title: 'Key points not found',
      detail:
        'The pose cache did not show a clear setup-to-top pattern. Use a face-on or down-the-line swing that includes address and the finish.',
    });
    return advice;
  }

  if (address) {
    const head = abs(address.metrics.headSwayTowardLeadPct);
    const hip = abs(address.metrics.hipSwayTowardLeadPct);
    if (head !== null && head > 22) {
      advice.push({
        id: 'address-head',
        level: 'flag',
        title: 'Head is off the midline at address',
        detail: faceOn
          ? `At setup the head sits ${head.toFixed(0)}% of stance off the ankle line. Start quieter over mid-stance so the swing does not chase the head.`
          : `At setup the head is ${head.toFixed(0)}% of stance off the ball line. Get the cranium over the ball before you take it away.`,
      });
    }
    if (
      address.metrics.shoulderTiltDeg !== null &&
      address.metrics.pelvicTiltDeg !== null &&
      Math.sign(address.metrics.shoulderTiltDeg) !== Math.sign(address.metrics.pelvicTiltDeg) &&
      Math.abs(address.metrics.shoulderTiltDeg) > 10 &&
      Math.abs(address.metrics.pelvicTiltDeg) > 8
    ) {
      advice.push({
        id: 'address-reverse-k',
        level: 'watch',
        title: 'Opposite shoulder and hip tilt at setup',
        detail:
          'Shoulders and pelvis tilt different ways at address (a reverse-K look). Level the girdles, then add your intended trail-side tilt.',
      });
    }
    if (address.metrics.trailKneeDeg !== null && address.metrics.trailKneeDeg > 174) {
      advice.push({
        id: 'address-trail-knee',
        level: 'watch',
        title: 'Trail knee locked at address',
        detail: 'A soft flex in the trail knee at setup usually lets the pelvis turn instead of slide.',
      });
    }
    if (faceOn && address.metrics.stanceToShoulderRatio !== null) {
      if (address.metrics.stanceToShoulderRatio < 0.85) {
        advice.push({
          id: 'address-narrow',
          level: 'info',
          title: 'Narrow stance',
          detail: `Ankles are ${(address.metrics.stanceToShoulderRatio * 100).toFixed(0)}% of shoulder width. A driver stance at or just outside the shoulders is more stable.`,
        });
      } else if (address.metrics.stanceToShoulderRatio > 1.7) {
        advice.push({
          id: 'address-wide',
          level: 'info',
          title: 'Very wide stance',
          detail: `Ankles are ${(address.metrics.stanceToShoulderRatio * 100).toFixed(0)}% of shoulder width. Stable, but it can limit pelvic turn.`,
        });
      }
    }
    if (hip !== null && hip > 16) {
      advice.push({
        id: 'address-hips',
        level: 'watch',
        title: 'Pelvis already off the line at setup',
        detail: faceOn
          ? `Hips are ${hip.toFixed(0)}% of stance from the midline before the club moves. Square the pelvis over the ankles at address.`
          : `Hips are ${hip.toFixed(0)}% of stance from the ball line at setup. That often shows up as early extension later.`,
      });
    }
  }

  if (address && top) {
    const headShift =
      address.metrics.headSwayTowardLeadPct !== null && top.metrics.headSwayTowardLeadPct !== null
        ? top.metrics.headSwayTowardLeadPct - address.metrics.headSwayTowardLeadPct
        : null;
    if (headShift !== null && Math.abs(headShift) > 18) {
      advice.push({
        id: 'backswing-head',
        level: 'watch',
        title: 'Head moves a lot going back',
        detail: `Head offset changes by ${Math.abs(headShift).toFixed(0)}% of stance from address to the top. Keep the head more centered while the shoulders turn.`,
      });
    }
    if (top.metrics.leadElbowDeg !== null && top.metrics.leadElbowDeg < 140) {
      advice.push({
        id: 'top-lead-arm',
        level: 'info',
        title: 'Lead arm quite bent at the top',
        detail: 'A wide lead arm at the top usually stores more width. Confirm that the bend is a choice, not a collapse.',
      });
    }
  }

  if (address && impact) {
    const hipSlide =
      address.metrics.hipSwayTowardLeadPct !== null && impact.metrics.hipSwayTowardLeadPct !== null
        ? impact.metrics.hipSwayTowardLeadPct - address.metrics.hipSwayTowardLeadPct
        : null;
    const headSlide =
      address.metrics.headSwayTowardLeadPct !== null && impact.metrics.headSwayTowardLeadPct !== null
        ? impact.metrics.headSwayTowardLeadPct - address.metrics.headSwayTowardLeadPct
        : null;
    if (hipSlide !== null && hipSlide > 18) {
      advice.push({
        id: 'impact-slide',
        level: 'flag',
        title: 'Pelvis slides toward the lead side into impact',
        detail: faceOn
          ? `Hips move ${hipSlide.toFixed(0)}% of stance toward the lead side from setup to impact. Turn the belt buckle more, slide it less.`
          : `Hips move ${hipSlide.toFixed(0)}% of stance toward the lead side into impact. That slide often pairs with losing posture.`,
      });
    }
    if (headSlide !== null && Math.abs(headSlide) > 20) {
      advice.push({
        id: 'impact-head',
        level: 'watch',
        title: 'Head does not stay with the setup',
        detail: `Head offset changes by ${Math.abs(headSlide).toFixed(0)}% of stance from address to impact. Try to keep the head where you started and let the body rotate under it.`,
      });
    }
    if (impact.metrics.leadKneeDeg !== null && impact.metrics.leadKneeDeg < 125) {
      advice.push({
        id: 'impact-lead-knee',
        level: 'watch',
        title: 'Lead knee very bent at impact',
        detail: 'A collapsing lead knee at impact often dumps the trail shoulder and the club behind you. Post into a firmer lead side.',
      });
    }
    if (
      !faceOn &&
      address.metrics.lateralTrunkFlexionDeg !== null &&
      impact.metrics.lateralTrunkFlexionDeg !== null &&
      Math.abs(impact.metrics.lateralTrunkFlexionDeg) + 8 < Math.abs(address.metrics.lateralTrunkFlexionDeg)
    ) {
      advice.push({
        id: 'impact-early-ext',
        level: 'watch',
        title: 'Spine stands up into impact',
        detail:
          'The torso is more vertical at impact than at address. That is a common early-extension pattern down the line — keep trail-hip depth through the ball.',
      });
    }
  }

  if (finish && top && finish.time <= top.time + 0.12) {
    advice.push({
      id: 'no-finish',
      level: 'info',
      title: 'Finish not separated from the top',
      detail: 'The clip may cut off before a full follow-through. Include the finish so the report can judge balance.',
    });
  }

  if (advice.length === 0) {
    advice.push({
      id: 'solid',
      level: 'info',
      title: faceOn ? 'Face-on key points look orderly' : 'Down-the-line key points look orderly',
      detail: faceOn
        ? 'Address, top, and impact stay in a reasonable frontal window. This is still 2D — confirm the same swing down the line.'
        : 'Address, top, and impact stay in a reasonable sagittal window. This is still 2D — confirm the same swing face-on.',
    });
  }

  return advice;
}

export function formatEventTime(time: number): string {
  const m = Math.floor(time / 60);
  const s = time - m * 60;
  return m > 0 ? `${m}:${s.toFixed(2).padStart(5, '0')}` : `${s.toFixed(2)}s`;
}
