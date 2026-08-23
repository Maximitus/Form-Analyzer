import type {GolfPhase} from './types';

export interface PhaseInferenceInput {
  wristElevation: number | null;
  hipSwayTowardLeadPct: number | null;
  peakWristElevation: number;
  prevWristElevation: number | null;
  prevPhase: GolfPhase;
  hasAddress: boolean;
}

/**
 * Coarse swing-phase guess from wrist height and pelvic sway.
 * Live HUD hint, not a full event detector.
 */
export function inferGolfPhase(input: PhaseInferenceInput): GolfPhase {
  const {wristElevation: elev, hipSwayTowardLeadPct: hip, peakWristElevation: peak} = input;
  if (elev === null) return input.prevPhase === 'unknown' ? 'unknown' : input.prevPhase;

  const rising = input.prevWristElevation !== null && elev - input.prevWristElevation > 0.04;
  const falling = input.prevWristElevation !== null && input.prevWristElevation - elev > 0.04;
  const prev = input.prevPhase;

  if (elev >= 1.05 || (peak >= 0.85 && elev >= peak * 0.92 && elev >= 0.8)) {
    return 'top';
  }

  if (prev === 'top' && falling) return 'downswing';
  if (prev === 'downswing' && falling) return 'downswing';
  if (prev === 'downswing' && elev < 0.5) {
    if ((hip ?? 0) > 4) return 'impact';
    return 'impact';
  }
  if (prev === 'impact') {
    if (rising || elev > 0.55) return 'follow-through';
    return 'impact';
  }
  if (prev === 'follow-through') {
    if (elev < 0.3 && !rising) return 'address';
    return 'follow-through';
  }

  if (elev < 0.32) {
    return input.hasAddress || prev === 'address' || prev === 'unknown' ? 'address' : prev;
  }

  if (rising || (elev > 0.4 && prev !== 'downswing')) {
    return 'backswing';
  }

  if (falling && elev > 0.4) return 'downswing';

  return prev === 'unknown' ? 'address' : prev;
}
