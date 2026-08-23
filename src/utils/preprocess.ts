import type { LetterboxMeta } from '../types/pose';
import { IMAGENET_MEAN, IMAGENET_STD } from '../types/pose';

export function computeLetterbox(
  srcWidth: number,
  srcHeight: number,
  dstWidth: number,
  dstHeight: number,
): LetterboxMeta {
  if (srcWidth <= 0 || srcHeight <= 0) {
    throw new Error('Source dimensions must be positive');
  }
  const scale = Math.min(dstWidth / srcWidth, dstHeight / srcHeight);
  const scaledW = srcWidth * scale;
  const scaledH = srcHeight * scale;
  return {
    scale,
    padX: (dstWidth - scaledW) / 2,
    padY: (dstHeight - scaledH) / 2,
    srcWidth,
    srcHeight,
    dstWidth,
    dstHeight,
  };
}

export function mapModelToSource(
  x: number,
  y: number,
  meta: LetterboxMeta,
): { x: number; y: number } {
  return {
    x: (x - meta.padX) / meta.scale,
    y: (y - meta.padY) / meta.scale,
  };
}

export function mapSourceToModel(
  x: number,
  y: number,
  meta: LetterboxMeta,
): { x: number; y: number } {
  return {
    x: x * meta.scale + meta.padX,
    y: y * meta.scale + meta.padY,
  };
}

/**
 * Convert packed RGBA ImageData into an NCHW Float32 tensor with ImageNet
 * mean/std normalization (RGB, 0–255 domain — MMPose / RTMPose convention).
 */
export function rgbaToNchwFloat32(
  rgba: Uint8ClampedArray | Uint8Array,
  height: number,
  width: number,
  mean: readonly [number, number, number] = IMAGENET_MEAN,
  std: readonly [number, number, number] = IMAGENET_STD,
): Float32Array {
  const plane = height * width;
  const out = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const src = i * 4;
    out[i] = (rgba[src] - mean[0]) / std[0];
    out[plane + i] = (rgba[src + 1] - mean[1]) / std[1];
    out[2 * plane + i] = (rgba[src + 2] - mean[2]) / std[2];
  }
  return out;
}
