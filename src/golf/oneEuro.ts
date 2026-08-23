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

export class OneEuroFilter {
  private readonly xFilter = new LowPassFilter();
  private readonly dxFilter = new LowPassFilter();
  private prevValue = 0;
  private prevTime: number | null = null;

  constructor(
    private readonly minCutoff = 1.2,
    private readonly beta = 0.4,
    private readonly dCutoff = 1,
  ) {}

  reset(): void {
    this.xFilter.reset();
    this.dxFilter.reset();
    this.prevValue = 0;
    this.prevTime = null;
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
    const edx = this.dxFilter.filter(dx, smoothingFactor(dt, this.dCutoff));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    const filtered = this.xFilter.filter(value, smoothingFactor(dt, cutoff));

    this.prevTime = timestampSeconds;
    this.prevValue = value;
    return filtered;
  }
}
