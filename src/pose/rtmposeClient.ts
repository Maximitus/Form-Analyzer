import { COCO_KEYPOINT_COUNT, COCO_KEYPOINT_NAMES, RTMPOSE_MODELS } from '../types/pose';
import type { ClubHeadEstimate, Keypoint2D, PoseFrame, RtmposeVariant } from '../types/pose';
import type { WorkerInboundMessage, WorkerOutboundMessage } from './protocol';

export interface RtmposeClientOptions {
  variant?: RtmposeVariant;
  wasmPaths?: string;
  onReady?: (info: { executionProvider: string; inputWidth: number; inputHeight: number }) => void;
  onPose?: (pose: PoseFrame) => void;
  onError?: (message: string) => void;
}

function withBaseUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) return url;
  const base = import.meta.env.BASE_URL ?? '/';
  const prefix = base.endsWith('/') ? base : `${base}/`;
  return `${prefix}${url.replace(/^\//, '')}`;
}

function defaultWasmPaths(): string {
  const origin = self.location?.origin ?? '';
  return `${origin}${withBaseUrl('ort/')}`;
}

function poseFromMessage(message: Extract<WorkerOutboundMessage, {type: 'pose'}>): PoseFrame {
  const keypoints: Keypoint2D[] = [];
  for (let i = 0; i < Math.min(COCO_KEYPOINT_COUNT, message.xs.length); i++) {
    keypoints.push({
      x: message.xs[i],
      y: message.ys[i],
      score: message.scores[i],
      name: COCO_KEYPOINT_NAMES[i],
    });
  }
  let clubHead: ClubHeadEstimate | undefined;
  if (message.clubHead && message.clubHead.length >= 6) {
    clubHead = {
      x: message.clubHead[0],
      y: message.clubHead[1],
      score: message.clubHead[2],
      gripX: message.clubHead[3],
      gripY: message.clubHead[4],
      method: message.clubHead[5] >= 0.5 ? 'image' : 'prior',
    };
  }
  return {
    frameId: message.frameId,
    mediaTime: message.mediaTime,
    inferenceMs: message.inferenceMs,
    width: message.width,
    height: message.height,
    keypoints,
    clubHead,
  };
}

export class RtmposeClient {
  private worker: Worker | null = null;
  private busy = false;
  private frameId = 0;
  private readonly options: RtmposeClientOptions;
  private readyPromise: Promise<void> | null = null;
  private readyResolve: (() => void) | null = null;
  private readyReject: ((err: Error) => void) | null = null;
  private inferWaiters = new Map<number, {resolve: (pose: PoseFrame) => void; reject: (err: Error) => void}>();

  constructor(options: RtmposeClientOptions = {}) {
    this.options = options;
  }

  get isBusy(): boolean {
    return this.busy;
  }

  async start(): Promise<void> {
    this.stop();
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });
    const variant = this.options.variant ?? 's';
    const spec = RTMPOSE_MODELS[variant];
    const worker = new Worker(new URL('../workers/rtmpose.worker.ts', import.meta.url), {
      type: 'module',
    });
    this.worker = worker;

    worker.onmessage = (event: MessageEvent<WorkerOutboundMessage>) => {
      const message = event.data;
      if (message.type === 'pose') {
        this.busy = false;
        const pose = poseFromMessage(message);
        const waiter = this.inferWaiters.get(message.frameId);
        if (waiter) {
          this.inferWaiters.delete(message.frameId);
          waiter.resolve(pose);
        }
        this.options.onPose?.(pose);
        return;
      }
      if (message.type === 'ready') {
        this.readyResolve?.();
        this.readyResolve = null;
        this.readyReject = null;
        this.options.onReady?.({
          executionProvider: message.executionProvider,
          inputWidth: message.inputWidth,
          inputHeight: message.inputHeight,
        });
        return;
      }
      if (message.type === 'error') {
        this.busy = false;
        const err = new Error(message.message);
        this.readyReject?.(err);
        for (const waiter of this.inferWaiters.values()) waiter.reject(err);
        this.inferWaiters.clear();
        this.options.onError?.(message.message);
      }
    };

    worker.onerror = (event) => {
      this.busy = false;
      const err = new Error(event.message || 'RTMPose worker crashed');
      this.readyReject?.(err);
      for (const waiter of this.inferWaiters.values()) waiter.reject(err);
      this.inferWaiters.clear();
      this.options.onError?.(event.message || 'RTMPose worker crashed');
    };

    const envOverride =
      variant === 's'
        ? import.meta.env.VITE_RTMPOSE_MODEL_S
        : import.meta.env.VITE_RTMPOSE_MODEL_M;

    const init: Extract<WorkerInboundMessage, { type: 'init' }> = {
      type: 'init',
      modelUrl: envOverride || withBaseUrl(spec.localUrl),
      fallbackUrl: spec.remoteUrl,
      wasmPaths: this.options.wasmPaths ?? defaultWasmPaths(),
      inputWidth: spec.inputWidth,
      inputHeight: spec.inputHeight,
      simccSplitRatio: spec.simccSplitRatio,
    };
    worker.postMessage(init);
    await this.readyPromise;
  }

  async inferOnce(
    bitmap: ImageBitmap,
    mediaTime: number,
    options: { trackClubHead?: boolean; leadIsLeft?: boolean } = {},
  ): Promise<PoseFrame> {
    if (!this.worker || !this.readyPromise) {
      bitmap.close();
      throw new Error('RTMPose worker is not started');
    }
    await this.readyPromise;
    const started = performance.now();
    while (this.busy) {
      if (performance.now() - started > 8000) {
        bitmap.close();
        throw new Error('RTMPose worker busy');
      }
      await new Promise((r) => setTimeout(r, 8));
    }
    const frameId = this.frameId + 1;
    return new Promise<PoseFrame>((resolve, reject) => {
      this.inferWaiters.set(frameId, {resolve, reject});
      const ok = this.submitFrame(bitmap, mediaTime, options);
      if (!ok) {
        this.inferWaiters.delete(frameId);
        reject(new Error('RTMPose dropped frame'));
      }
    });
  }

  /**
   * Transfer an ImageBitmap frame to the worker. Returns false if the worker
   * is still inferring the previous frame (backpressure).
   */
  submitFrame(
    bitmap: ImageBitmap,
    mediaTime: number,
    options: { trackClubHead?: boolean; leadIsLeft?: boolean } = {},
  ): boolean {
    if (!this.worker || this.busy) {
      bitmap.close();
      return false;
    }
    this.busy = true;
    const frameId = ++this.frameId;
    const message: Extract<WorkerInboundMessage, { type: 'frame' }> = {
      type: 'frame',
      frameId,
      mediaTime,
      bitmap,
      trackClubHead: options.trackClubHead,
      leadIsLeft: options.leadIsLeft,
    };
    this.worker.postMessage(message, [bitmap]);
    return true;
  }

  resetClub(): void {
    this.worker?.postMessage({ type: 'resetClub' } satisfies WorkerInboundMessage);
  }

  stop(): void {
    const err = new Error('RTMPose worker stopped');
    for (const waiter of this.inferWaiters.values()) waiter.reject(err);
    this.inferWaiters.clear();
    this.readyPromise = null;
    this.readyResolve = null;
    this.readyReject = null;
    if (this.worker) {
      this.worker.postMessage({ type: 'dispose' } satisfies WorkerInboundMessage);
      this.worker.terminate();
      this.worker = null;
    }
    this.busy = false;
  }
}
