import type { QualityReport, QualityWarning, RejectReason } from '@/types/domain';

export const QUALITY = {
  minSide: 64,
  warnSide: 160,
  blankStd: 3,
  darkMean: 12,
  brightMean: 248,
  blurSharpness: 12,
  analysisSize: 64,
} as const;

export interface LumaStats {
  mean: number;
  std: number;
  sharpness: number;
}

/** Luma statistics from a square grayscale buffer; sharpness = variance of a 4-neighbour Laplacian. */
export function lumaStats(luma: ArrayLike<number>, size: number): LumaStats {
  const n = size * size;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += luma[i] as number;
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i++) varSum += ((luma[i] as number) - mean) ** 2;
  const std = Math.sqrt(varSum / n);
  let lapSum = 0;
  let lapSq = 0;
  let cnt = 0;
  for (let y = 1; y < size - 1; y++) {
    for (let x = 1; x < size - 1; x++) {
      const i = y * size + x;
      const lap = 4 * (luma[i] as number) - (luma[i - 1] as number) - (luma[i + 1] as number) - (luma[i - size] as number) - (luma[i + size] as number);
      lapSum += lap;
      lapSq += lap * lap;
      cnt++;
    }
  }
  const lm = lapSum / cnt;
  return { mean, std, sharpness: lapSq / cnt - lm * lm };
}

export interface QualityVerdict {
  reject: { reason: RejectReason; detail: string } | null;
  report: QualityReport;
}

export function assessQuality(stats: LumaStats, width: number, height: number): QualityVerdict {
  const warnings: QualityWarning[] = [];
  const minSide = Math.min(width, height);
  if (minSide >= QUALITY.minSide && minSide < QUALITY.warnSide) warnings.push('low-resolution');
  if (stats.std >= QUALITY.blankStd && stats.sharpness < QUALITY.blurSharpness) warnings.push('blurry');
  const report: QualityReport = { meanLuma: stats.mean, stdLuma: stats.std, sharpness: stats.sharpness, warnings };
  let reject: QualityVerdict['reject'] = null;
  if (minSide < QUALITY.minSide) reject = { reason: 'too-small', detail: `${width}×${height}px is below the ${QUALITY.minSide}px minimum side.` };
  else if (stats.std < QUALITY.blankStd) reject = { reason: 'blank', detail: 'The image has almost no variation (blank or a single colour).' };
  else if (stats.mean < QUALITY.darkMean) reject = { reason: 'too-dark', detail: 'The image is almost completely black.' };
  else if (stats.mean > QUALITY.brightMean) reject = { reason: 'too-bright', detail: 'The image is almost completely white (over-exposed).' };
  return { reject, report };
}
