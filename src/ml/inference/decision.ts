import type { Prediction } from '@/types/ml';

export interface Decision {
  label: string | null;
  isUnknown: boolean;
  top: Prediction;
  margin: number;
  /** Normalised Shannon entropy in [0,1]; 1 = uniform (maximally uncertain). */
  entropy: number;
}

/**
 * Softmax outputs sum to 1 by construction, so they always "pick" a class, even for images
 * unlike anything in the dataset. They are also usually over-confident (not calibrated).
 * A threshold rejects low-confidence outputs; it cannot detect confidently wrong out-of-distribution inputs.
 */
export function decide(predictions: Prediction[], threshold: number): Decision {
  const sorted = [...predictions].sort((a, b) => b.probability - a.probability);
  const top = sorted[0] as Prediction;
  const second = sorted[1];
  const n = sorted.length;
  let h = 0;
  for (const p of sorted) if (p.probability > 0) h -= p.probability * Math.log(p.probability);
  const entropy = n > 1 ? h / Math.log(n) : 0;
  const isUnknown = top.probability < threshold;
  return { label: isUnknown ? null : top.className, isUnknown, top, margin: top.probability - (second?.probability ?? 0), entropy };
}
