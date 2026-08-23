/**
 * 1€ (One-Euro) Filter — Casiez, Roussel, Vogel (CHI 2012).
 *
 * Low-pass smoothing whose cutoff rises with signal speed, so jitter is
 * suppressed at rest while fast eccentric/concentric motion stays snappy.
 */

function smoothingFactor(deltaTime: number, cutoff: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / Math.max(deltaTime, 1e-6));
}

class LowPassFilter {
  private hatxPrev = 0;
  private initialized = false;

  reset(): void {
    this.hatxPrev = 0;
    this.initialized = false;
  }

  last(): number {
    return this.hatxPrev;
  }

  filter(value: number, alpha: number): number {
    if (!this.initialized) {
      this.initialized = true;
      this.hatxPrev = value;
      return value;
    }
    const hatx = alpha * value + (1 - alpha) * this.hatxPrev;
    this.hatxPrev = hatx;
    return hatx;
  }
}

export interface OneEuroParams {
  /** Minimum cutoff frequency (Hz). Higher = less lag, more jitter. */
  minCutoff: number;
  /** Speed coefficient. Higher = faster tracking of quick motion. */
  beta: number;
  /** Cutoff for the derivative (Hz). */
  dCutoff: number;
}

export const DEFAULT_ONE_EURO: OneEuroParams = {
  minCutoff: 1.0,
  beta: 0.3,
  dCutoff: 1.0,
};

export class OneEuroFilter {
  private readonly xFilter = new LowPassFilter();
  private readonly dxFilter = new LowPassFilter();
  private prevValue = 0;
  private prevTime: number | null = null;
  private readonly params: OneEuroParams;

  constructor(params: Partial<OneEuroParams> = {}) {
    this.params = { ...DEFAULT_ONE_EURO, ...params };
  }

  reset(): void {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.prevValue = 0;
    this.prevTime = null;
  }

  last(): number {
    return this.xFilter.last();
  }

  filter(value: number, timestampSeconds: number): number {
    if (this.prevTime === null) {
      this.prevTime = timestampSeconds;
      this.prevValue = value;
      this.xFilter.filter(value, 1);
      this.dxFilter.filter(0, 1);
      return value;
    }

    const dt = timestampSeconds - this.prevTime;
    if (dt <= 0) {
      return this.xFilter.filter(value, 1);
    }

    const dx = (value - this.prevValue) / dt;
    const edx = this.dxFilter.filter(dx, smoothingFactor(dt, this.params.dCutoff));
    const cutoff = this.params.minCutoff + this.params.beta * Math.abs(edx);
    const filtered = this.xFilter.filter(value, smoothingFactor(dt, cutoff));

    this.prevTime = timestampSeconds;
    this.prevValue = value;
    return filtered;
  }
}

export class KeypointOneEuroFilter {
  private readonly xFilters: OneEuroFilter[];
  private readonly yFilters: OneEuroFilter[];
  private readonly lastGoodX: Float32Array;
  private readonly lastGoodY: Float32Array;
  private readonly seen: Uint8Array;

  constructor(
    private readonly count: number,
    params: Partial<OneEuroParams> = {},
  ) {
    this.xFilters = Array.from({ length: count }, () => new OneEuroFilter(params));
    this.yFilters = Array.from({ length: count }, () => new OneEuroFilter(params));
    this.lastGoodX = new Float32Array(count);
    this.lastGoodY = new Float32Array(count);
    this.seen = new Uint8Array(count);
  }

  reset(): void {
    for (const f of this.xFilters) f.reset();
    for (const f of this.yFilters) f.reset();
    this.lastGoodX.fill(0);
    this.lastGoodY.fill(0);
    this.seen.fill(0);
  }

  /**
   * Smooth x/y independently. Low-confidence detections keep the previous
   * filtered position so missing joints do not snap.
   */
  apply(
    xs: Float32Array,
    ys: Float32Array,
    scores: Float32Array,
    timestampSeconds: number,
    minScore = 0.25,
  ): { xs: Float32Array; ys: Float32Array; scores: Float32Array } {
    const outX = new Float32Array(this.count);
    const outY = new Float32Array(this.count);
    const outS = new Float32Array(this.count);

    for (let i = 0; i < this.count; i++) {
      const score = scores[i] ?? 0;
      outS[i] = score;

      if (score >= minScore) {
        outX[i] = this.xFilters[i].filter(xs[i], timestampSeconds);
        outY[i] = this.yFilters[i].filter(ys[i], timestampSeconds);
        this.lastGoodX[i] = outX[i];
        this.lastGoodY[i] = outY[i];
        this.seen[i] = 1;
      } else if (this.seen[i]) {
        outX[i] = this.lastGoodX[i];
        outY[i] = this.lastGoodY[i];
      } else {
        outX[i] = xs[i];
        outY[i] = ys[i];
      }
    }

    return { xs: outX, ys: outY, scores: outS };
  }
}
