import type { ClassMetrics, EvaluationReport, MisclassifiedSample } from '@/types/ml';

export interface EvalInput {
  classIds: string[];
  classNames: string[];
  sampleIds: string[];
  /** true class index per sample */
  yTrue: ArrayLike<number>;
  /** flattened [n * numClasses] probabilities */
  probs: ArrayLike<number>;
  maxMisclassified?: number;
}

export function argmax(probs: ArrayLike<number>, offset: number, len: number): number {
  let best = 0;
  let bv = -Infinity;
  for (let k = 0; k < len; k++) {
    const v = probs[offset + k] as number;
    if (v > bv) {
      bv = v;
      best = k;
    }
  }
  return best;
}

export function buildEvaluationReport(input: EvalInput): EvaluationReport {
  const c = input.classIds.length;
  const n = input.sampleIds.length;
  const confusion = Array.from({ length: c }, () => new Array<number>(c).fill(0));
  const wrong: MisclassifiedSample[] = [];
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const actual = input.yTrue[i] as number;
    const pred = argmax(input.probs, i * c, c);
    (confusion[actual] as number[])[pred]! += 1;
    if (actual === pred) correct++;
    else {
      wrong.push({
        sampleId: input.sampleIds[i] as string,
        trueClassId: input.classIds[actual] as string,
        predictedClassId: input.classIds[pred] as string,
        probability: input.probs[i * c + pred] as number,
        trueProbability: input.probs[i * c + actual] as number,
      });
    }
  }
  const perClass: ClassMetrics[] = input.classIds.map((classId, k) => {
    const tp = (confusion[k] as number[])[k] as number;
    const support = (confusion[k] as number[]).reduce((a, b) => a + b, 0);
    const predicted = confusion.reduce((a, row) => a + (row[k] as number), 0);
    const precision = predicted ? tp / predicted : 0;
    const recall = support ? tp / support : 0;
    const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
    return { classId, className: input.classNames[k] as string, precision, recall, f1, support };
  });
  const present = perClass.filter((m) => m.support > 0);
  const macroF1 = present.length ? present.reduce((a, m) => a + m.f1, 0) / present.length : 0;
  const weightedF1 = n ? perClass.reduce((a, m) => a + m.f1 * m.support, 0) / n : 0;
  wrong.sort((a, b) => b.probability - a.probability);
  return {
    sampleCount: n,
    accuracy: n ? correct / n : 0,
    macroF1,
    weightedF1,
    perClass,
    confusion,
    classIds: input.classIds,
    classNames: input.classNames,
    misclassified: wrong.slice(0, input.maxMisclassified ?? 24),
  };
}
