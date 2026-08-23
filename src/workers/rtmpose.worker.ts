/// <reference types="vite/client" />

import * as ort from 'onnxruntime-web/webgpu';
import { decodeSimcc, identifySimccOutputs } from '../utils/simcc';
import { computeLetterbox, mapModelToSource, rgbaToNchwFloat32 } from '../utils/preprocess';
import { estimateClubHead, landmarksFromCoco } from '../utils/clubHead';
import { COCO_KEYPOINT_COUNT } from '../types/pose';
import type { Point2D } from '../types/pose';
import type { WorkerInboundMessage, WorkerOutboundMessage } from '../pose/protocol';

interface SessionState {
  session: ort.InferenceSession;
  inputName: string;
  inputWidth: number;
  inputHeight: number;
  simccSplitRatio: number;
  xName: string;
  yName: string;
  executionProvider: string;
  canvas: OffscreenCanvas;
  ctx: OffscreenCanvasRenderingContext2D;
  searchCanvas: OffscreenCanvas;
  searchCtx: OffscreenCanvasRenderingContext2D;
  prevClub: { x: number; y: number } | null;
}

let state: SessionState | null = null;

function scalePoint(point: Point2D | null, scale: number): Point2D | null {
  return point ? { x: point.x * scale, y: point.y * scale } : null;
}

function post(message: WorkerOutboundMessage, transfer?: Transferable[]): void {
  if (transfer && transfer.length > 0) {
    self.postMessage(message, { transfer });
  } else {
    self.postMessage(message);
  }
}

function configureOrt(wasmPaths: string): void {
  ort.env.wasm.wasmPaths = wasmPaths.endsWith('/') ? wasmPaths : `${wasmPaths}/`;
  const isolated = Boolean((self as unknown as {crossOriginIsolated?: boolean}).crossOriginIsolated);
  // Threaded WASM needs COOP/COEP (SharedArrayBuffer). Stay single-threaded unless isolated.
  // Do not request the JSEP (~26MB) binary; Cloudflare Workers assets cap files at 25 MiB.
  ort.env.wasm.numThreads = isolated ? Math.min(4, self.navigator?.hardwareConcurrency ?? 1) : 1;
  ort.env.wasm.simd = true;
  // Already running inside a dedicated worker; nested ORT proxy workers are unnecessary.
  ort.env.wasm.proxy = false;
}

async function fetchModelBuffer(url: string): Promise<ArrayBuffer> {
  try {
    const cache = await caches.open('rtmpose-onnx-v1');
    const cached = await cache.match(url);
    if (cached) {
      return cached.arrayBuffer();
    }
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} fetching ${url}`);
    }
    await cache.put(url, response.clone());
    return response.arrayBuffer();
  } catch {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status} fetching ${url}`);
    }
    return response.arrayBuffer();
  }
}

async function loadModelBytes(primary: string, fallback?: string): Promise<ArrayBuffer> {
  try {
    return await fetchModelBuffer(primary);
  } catch (first) {
    if (!fallback || fallback === primary) throw first;
    return fetchModelBuffer(fallback);
  }
}

async function createSession(model: ArrayBuffer): Promise<{
  session: ort.InferenceSession;
  executionProvider: string;
}> {
  try {
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
    });
    const ep =
      (session as unknown as { handler?: { backendHint?: string } }).handler?.backendHint ??
      'webgpu';
    return { session, executionProvider: ep };
  } catch {
    const session = await ort.InferenceSession.create(model, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return { session, executionProvider: 'wasm' };
  }
}

function tensorShape(meta: ort.InferenceSession.ValueMetadata | undefined): number[] {
  if (!meta || !meta.isTensor) return [];
  return meta.shape.map((d) => (typeof d === 'number' ? d : Number(d) || 0));
}

function readInputLayout(
  session: ort.InferenceSession,
  fallbackWidth: number,
  fallbackHeight: number,
): { name: string; width: number; height: number } {
  const name = session.inputNames[0];
  const meta =
    session.inputMetadata.find((item) => item.name === name) ?? session.inputMetadata[0];
  const dims = tensorShape(meta);
  // NCHW: [N, C, H, W]  or NHWC: [N, H, W, C]
  if (dims.length === 4 && dims[1] === 3) {
    return { name, height: dims[2] || fallbackHeight, width: dims[3] || fallbackWidth };
  }
  if (dims.length === 4 && dims[3] === 3) {
    return { name, height: dims[1] || fallbackHeight, width: dims[2] || fallbackWidth };
  }
  return { name, width: fallbackWidth, height: fallbackHeight };
}

async function init(message: Extract<WorkerInboundMessage, { type: 'init' }>): Promise<void> {
  configureOrt(message.wasmPaths);
  const bytes = await loadModelBytes(message.modelUrl, message.fallbackUrl);
  const { session, executionProvider } = await createSession(bytes);
  const layout = readInputLayout(session, message.inputWidth, message.inputHeight);

  const identified = identifySimccOutputs(
    session.outputMetadata.map((meta) => ({
      name: meta.name,
      dims: tensorShape(meta),
    })),
    layout.width,
    layout.height,
    message.simccSplitRatio,
  );

  const canvas = new OffscreenCanvas(layout.width, layout.height);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) {
    throw new Error('OffscreenCanvas 2D context is unavailable');
  }
  const searchCanvas = new OffscreenCanvas(384, 384);
  const searchCtx = searchCanvas.getContext('2d', { willReadFrequently: true });
  if (!searchCtx) {
    throw new Error('Club-search OffscreenCanvas 2D context is unavailable');
  }

  state = {
    session,
    inputName: layout.name,
    inputWidth: layout.width,
    inputHeight: layout.height,
    simccSplitRatio: message.simccSplitRatio,
    xName: identified.xName,
    yName: identified.yName,
    executionProvider,
    canvas,
    ctx,
    searchCanvas,
    searchCtx,
    prevClub: null,
  };

  post({
    type: 'ready',
    executionProvider,
    inputName: layout.name,
    inputWidth: layout.width,
    inputHeight: layout.height,
  });
}

async function inferFrame(
  message: Extract<WorkerInboundMessage, { type: 'frame' }>,
): Promise<void> {
  if (!state) {
    message.bitmap.close();
    throw new Error('RTMPose session is not initialized');
  }

  const { bitmap, frameId, mediaTime } = message;
  const srcWidth = bitmap.width;
  const srcHeight = bitmap.height;
  const t0 =
    typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();

  const meta = computeLetterbox(srcWidth, srcHeight, state.inputWidth, state.inputHeight);
  state.ctx.fillStyle = '#000000';
  state.ctx.fillRect(0, 0, state.inputWidth, state.inputHeight);
  state.ctx.drawImage(
    bitmap,
    meta.padX,
    meta.padY,
    srcWidth * meta.scale,
    srcHeight * meta.scale,
  );

  let searchScale = 1;
  let searchPixels: ImageData | null = null;
  if (message.trackClubHead) {
    searchScale = Math.min(1, 384 / srcWidth, 384 / srcHeight);
    const sw = Math.max(1, Math.round(srcWidth * searchScale));
    const sh = Math.max(1, Math.round(srcHeight * searchScale));
    if (state.searchCanvas.width !== sw || state.searchCanvas.height !== sh) {
      state.searchCanvas.width = sw;
      state.searchCanvas.height = sh;
    }
    state.searchCtx.drawImage(bitmap, 0, 0, sw, sh);
    searchPixels = state.searchCtx.getImageData(0, 0, sw, sh);
  }
  bitmap.close();

  const imageData = state.ctx.getImageData(0, 0, state.inputWidth, state.inputHeight);
  const nchw = rgbaToNchwFloat32(imageData.data, state.inputHeight, state.inputWidth);
  const input = new ort.Tensor('float32', nchw, [1, 3, state.inputHeight, state.inputWidth]);

  const results = await state.session.run({ [state.inputName]: input });
  const simccX = results[state.xName];
  const simccY = results[state.yName];
  if (!simccX || !simccY) {
    throw new Error(`Missing SimCC outputs (${state.xName}, ${state.yName})`);
  }

  const decoded = decodeSimcc(
    simccX.data as Float32Array,
    simccX.dims,
    simccY.data as Float32Array,
    simccY.dims,
    { splitRatio: state.simccSplitRatio },
  );

  const count = Math.min(COCO_KEYPOINT_COUNT, decoded.xs.length);
  const xs = new Float32Array(count);
  const ys = new Float32Array(count);
  const scores = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const mapped = mapModelToSource(decoded.xs[i], decoded.ys[i], meta);
    xs[i] = mapped.x;
    ys[i] = mapped.y;
    scores[i] = decoded.scores[i];
  }

  let clubHead: Float32Array | undefined;
  if (searchPixels) {
    const landmarks = landmarksFromCoco(xs, ys, scores, message.leadIsLeft !== false);
    const searchLandmarks = {
      leadWrist: scalePoint(landmarks.leadWrist, searchScale),
      trailWrist: scalePoint(landmarks.trailWrist, searchScale),
      midHip: scalePoint(landmarks.midHip, searchScale),
      midShoulder: scalePoint(landmarks.midShoulder, searchScale),
      head: scalePoint(landmarks.head, searchScale),
      midAnkle: scalePoint(landmarks.midAnkle, searchScale),
    };
    const prev = state.prevClub
      ? { x: state.prevClub.x * searchScale, y: state.prevClub.y * searchScale }
      : null;
    const found = estimateClubHead(
      searchPixels.data,
      searchPixels.width,
      searchPixels.height,
      searchLandmarks,
      prev,
    );
    if (found) {
      const x = found.x / searchScale;
      const y = found.y / searchScale;
      state.prevClub = { x, y };
      clubHead = Float32Array.from([
        x,
        y,
        found.score,
        found.gripX / searchScale,
        found.gripY / searchScale,
        found.method === 'image' ? 1 : 0,
      ]);
    }
  }

  const t1 =
    typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now();

  const transfer: Transferable[] = [xs.buffer, ys.buffer, scores.buffer];
  if (clubHead) transfer.push(clubHead.buffer);

  post(
    {
      type: 'pose',
      frameId,
      mediaTime,
      inferenceMs: t1 - t0,
      width: srcWidth,
      height: srcHeight,
      xs,
      ys,
      scores,
      clubHead,
    },
    transfer,
  );
}

async function dispose(): Promise<void> {
  if (state) {
    await state.session.release();
    state.prevClub = null;
    state = null;
  }
}

self.onmessage = (event: MessageEvent<WorkerInboundMessage>) => {
  const message = event.data;
  void (async () => {
    try {
      switch (message.type) {
        case 'init':
          await init(message);
          break;
        case 'frame':
          await inferFrame(message);
          break;
        case 'dispose':
          await dispose();
          break;
        case 'resetClub':
          if (state) state.prevClub = null;
          break;
        default:
          break;
      }
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      post({ type: 'error', message: err.message });
    }
  })();
};
