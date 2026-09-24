import type { PreprocessingSpec } from '@/types/ml';
import { CHANNELS } from './spec';

type AnyCanvas = OffscreenCanvas | HTMLCanvasElement;
type AnyCtx = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;

const canvasCache = new Map<number, { canvas: AnyCanvas; ctx: AnyCtx }>();

function getCanvas(size: number): { canvas: AnyCanvas; ctx: AnyCtx } {
  const hit = canvasCache.get(size);
  if (hit) return hit;
  const canvas: AnyCanvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(size, size) : Object.assign(document.createElement('canvas'), { width: size, height: size });
  const ctx = canvas.getContext('2d', { willReadFrequently: true }) as AnyCtx | null;
  if (!ctx) throw new Error('2D canvas context is unavailable in this environment.');
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const entry = { canvas, ctx };
  canvasCache.set(size, entry);
  return entry;
}

/** Deterministic resize of an ImageBitmap to `spec.inputSize`² RGB bytes. */
export function bitmapToPixels(bitmap: ImageBitmap, spec: PreprocessingSpec): Uint8Array {
  const size = spec.inputSize;
  const { ctx } = getCanvas(size);
  ctx.fillStyle = 'rgb(128,128,128)';
  ctx.fillRect(0, 0, size, size);
  const w = bitmap.width;
  const h = bitmap.height;
  if (spec.resize === 'stretch') {
    ctx.drawImage(bitmap, 0, 0, w, h, 0, 0, size, size);
  } else if (spec.resize === 'center-crop') {
    const s = Math.min(w, h);
    ctx.drawImage(bitmap, (w - s) / 2, (h - s) / 2, s, s, 0, 0, size, size);
  } else {
    const scale = size / Math.max(w, h);
    const dw = Math.max(1, Math.round(w * scale));
    const dh = Math.max(1, Math.round(h * scale));
    ctx.drawImage(bitmap, 0, 0, w, h, Math.floor((size - dw) / 2), Math.floor((size - dh) / 2), dw, dh);
  }
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const out = new Uint8Array(size * size * CHANNELS);
  for (let i = 0, j = 0; i < rgba.length; i += 4) {
    out[j++] = rgba[i] as number;
    out[j++] = rgba[i + 1] as number;
    out[j++] = rgba[i + 2] as number;
  }
  return out;
}

export async function decodeBlob(blob: Blob): Promise<ImageBitmap> {
  return createImageBitmap(blob, { imageOrientation: 'from-image' });
}
