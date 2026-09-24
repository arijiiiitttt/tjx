import type { DatasetClass, DatasetIssue, ImageSample } from '@/types/domain';
import type { TrainingConfig } from '@/types/ml';

export interface ClassStats {
  classId: string;
  name: string;
  count: number;
  camera: number;
  upload: number;
  groups: number;
  lowQuality: number;
}

export interface DatasetAnalysis {
  total: number;
  classes: ClassStats[];
  imbalanceRatio: number;
  issues: DatasetIssue[];
  blocking: boolean;
  needsAcknowledgement: boolean;
}

export const MIN_SAMPLES_PER_CLASS = 5;
export const RECOMMENDED_TRANSFER = 30;
export const RECOMMENDED_SCRATCH = 150;

export function classStats(classes: DatasetClass[], samples: ImageSample[]): ClassStats[] {
  return classes.map((c) => {
    const own = samples.filter((s) => s.classId === c.id);
    return {
      classId: c.id,
      name: c.name,
      count: own.length,
      camera: own.filter((s) => s.source === 'camera').length,
      upload: own.filter((s) => s.source === 'upload').length,
      groups: new Set(own.map((s) => s.groupId)).size,
      lowQuality: own.filter((s) => s.quality.warnings.length > 0).length,
    };
  });
}

export function analyzeDataset(classes: DatasetClass[], samples: ImageSample[], config: Pick<TrainingConfig, 'mode' | 'validationSplit'>): DatasetAnalysis {
  const stats = classStats(classes, samples);
  const issues: DatasetIssue[] = [];
  const used = stats.filter((s) => s.count > 0);
  const recommended = config.mode === 'scratch' ? RECOMMENDED_SCRATCH : RECOMMENDED_TRANSFER;

  if (used.length < 2) {
    issues.push({ severity: 'error', code: 'few-classes', message: `Training needs at least 2 classes with samples (found ${used.length}).` });
  }
  for (const s of stats) {
    if (s.count === 0 && used.length >= 2) {
      issues.push({ severity: 'warning', code: 'empty-class', classId: s.classId, message: `"${s.name}" has no samples and will be excluded from the model.` });
    } else if (s.count > 0 && s.count < MIN_SAMPLES_PER_CLASS) {
      issues.push({ severity: 'error', code: 'too-few', classId: s.classId, message: `"${s.name}" has ${s.count} samples; at least ${MIN_SAMPLES_PER_CLASS} are required to split and validate.` });
    } else if (s.count > 0 && s.count < recommended) {
      issues.push({ severity: 'warning', code: 'small-class', classId: s.classId, message: `"${s.name}" has ${s.count} samples; ${recommended}+ recommended for ${config.mode === 'scratch' ? 'training from scratch' : 'transfer learning'}.` });
    }
    if (s.count >= MIN_SAMPLES_PER_CLASS && s.groups === 1 && s.camera === s.count) {
      issues.push({ severity: 'warning', code: 'single-burst', classId: s.classId, message: `All "${s.name}" samples come from one camera burst. Validation frames will look very similar to training frames, so accuracy will be optimistic. Capture in several sessions, angles and backgrounds.` });
    }
  }

  const counts = used.map((s) => s.count);
  const imbalanceRatio = counts.length ? Math.max(...counts) / Math.max(1, Math.min(...counts)) : 1;
  if (used.length >= 2 && imbalanceRatio >= 2) {
    const detail = used.map((s) => `${s.name}: ${s.count}`).join(', ');
    issues.push({ severity: 'warning', code: 'imbalance', message: `Class imbalance detected (${imbalanceRatio.toFixed(1)}×) — ${detail}. Class-balanced loss weights are recommended.` });
  }
  const total = samples.length;
  if (config.mode === 'scratch' && total < 200 && used.length >= 2) {
    issues.push({ severity: 'warning', code: 'scratch-small', message: `Training a CNN from scratch on ${total} images will overfit. Expect a large gap between training and validation accuracy; transfer learning is the practical choice.` });
  }
  const lq = stats.reduce((a, s) => a + s.lowQuality, 0);
  if (lq > 0) issues.push({ severity: 'info', code: 'quality', message: `${lq} sample(s) have quality warnings (low resolution, blur or near-duplicate).` });

  return {
    total,
    classes: stats,
    imbalanceRatio,
    issues,
    blocking: issues.some((i) => i.severity === 'error'),
    needsAcknowledgement: issues.some((i) => i.severity === 'warning'),
  };
}

/** Short, content-derived fingerprint of the dataset (ids + labels), shown as the dataset version. */
export function datasetFingerprint(samples: ImageSample[]): string {
  const key = samples.map((s) => `${s.id}:${s.classId}:${s.sha256.slice(0, 8)}`).sort().join('|');
  let h1 = 0xdeadbeef;
  let h2 = 0x41c6ce57;
  for (let i = 0; i < key.length; i++) {
    const ch = key.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  return (h1 >>> 0).toString(16).padStart(8, '0');
}
