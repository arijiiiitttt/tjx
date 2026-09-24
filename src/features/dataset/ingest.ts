import { dHashFromLuma, sha256Hex } from '@/ml/preprocessing/hash';
import { assessQuality, lumaStats, QUALITY } from '@/ml/preprocessing/quality';
import type { QualityReport, RejectReason } from '@/types/domain';

export const MAX_FILE_BYTES = 25 * 1024 * 1024;
export const THUMB_SIZE = 128;

export interface AnalyzedImage {
  ok: true;
  width: number;
  height: number;
  mimeType: string;
  size: number;
  sha256: string;
  phash: string;
  quality: QualityReport;
  thumb: Blob;
}
export interface RejectedImage {
  ok: false;
  reason: RejectReason;
  detail: string;
}

function makeCanvas(w: number, h: number): OffscreenCanvas | HTMLCanvasElement {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(w, h);
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}
type Ctx2D = OffscreenCanvasRenderingContext2D | CanvasRenderingContext2D;
const ctxOf = (c: OffscreenCanvas | HTMLCanvasElement) => c.getContext('2d', { willReadFrequently: true }) as Ctx2D;

function lumaOf(canvasCtx: Ctx2D, w: number, h: number): Float32Array {
  const d = canvasCtx.getImageData(0, 0, w, h).data;
  const out = new Float32Array(w * h);
  for (let i = 0, j = 0; i < d.length; i += 4, j++) out[j] = 0.299 * (d[i] as number) + 0.587 * (d[i + 1] as number) + 0.114 * (d[i + 2] as number);
  return out;
}

async function toBlob(c: OffscreenCanvas | HTMLCanvasElement, type: string, quality: number): Promise<Blob> {
  if ('convertToBlob' in c) return c.convertToBlob({ type, quality });
  return new Promise((res, rej) => (c as HTMLCanvasElement).toBlob((b) => (b ? res(b) : rej(new Error('Canvas encoding failed.'))), type, quality));
}

/**
 * Validates and analyses one image WITHOUT modifying it. The original Blob is stored untouched by the caller;
 * this only derives quality statistics, hashes and a thumbnail.
 */
export async function analyzeImage(blob: Blob): Promise<AnalyzedImage | RejectedImage> {
  if (blob.size > MAX_FILE_BYTES) return { ok: false, reason: 'too-large', detail: `${(blob.size / 1048576).toFixed(1)} MB exceeds the ${MAX_FILE_BYTES / 1048576} MB limit.` };
  if (blob.type && !blob.type.startsWith('image/')) return { ok: false, reason: 'unsupported-type', detail: `"${blob.type}" is not an image type.` };
  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  } catch {
    return { ok: false, reason: 'corrupted', detail: 'The file could not be decoded as an image (corrupted or unsupported format).' };
  }
  try {
    const { width, height } = bmp;
    const n = QUALITY.analysisSize;
    const c = makeCanvas(n, n);
    const cx = ctxOf(c);
    cx.drawImage(bmp, 0, 0, width, height, 0, 0, n, n);
    const stats = lumaStats(lumaOf(cx, n, n), n);
    const verdict = assessQuality(stats, width, height);
    if (verdict.reject) return { ok: false, reason: verdict.reject.reason, detail: verdict.reject.detail };

    const hc = makeCanvas(9, 8);
    const hx = ctxOf(hc);
    hx.drawImage(bmp, 0, 0, width, height, 0, 0, 9, 8);
    const phash = dHashFromLuma(lumaOf(hx, 9, 8));

    const tc = makeCanvas(THUMB_SIZE, THUMB_SIZE);
    const tx = ctxOf(tc);
    const s = Math.min(width, height);
    tx.drawImage(bmp, (width - s) / 2, (height - s) / 2, s, s, 0, 0, THUMB_SIZE, THUMB_SIZE);
    const thumb = await toBlob(tc, 'image/jpeg', 0.8);
    return { ok: true, width, height, mimeType: blob.type || 'image/unknown', size: blob.size, sha256: await sha256Hex(blob), phash, quality: verdict.report, thumb };
  } finally {
    bmp.close();
  }
}
