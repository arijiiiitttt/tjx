import * as tf from '@tensorflow/tfjs';
import type { AugmentationConfig, PreprocessingSpec, TrainingConfig } from '@/types/ml';
import { mulberry32 } from '@/utils/rng';
import { augmentBatch, type AugmentCaps } from '../augmentation/augment';
import type { PreparedSample } from '../data/prepare';
import type { FeatureExtractor } from '../models/backbone';
import { CHANNELS, pixelsToUnit, unitToSpec } from '../preprocessing/spec';
import type { TrainData } from './engine';

export interface SplitIds {
  train: Set<string>;
  val: Set<string>;
}

const CHUNK = 32;

function stack(samples: PreparedSample[], from: number, to: number, size: number): Uint8Array {
  const per = size * size * CHANNELS;
  const buf = new Uint8Array((to - from) * per);
  for (let i = from; i < to; i++) buf.set((samples[i] as PreparedSample).pixels, (i - from) * per);
  return buf;
}

/**
 * Transfer learning data: the backbone is frozen, so embeddings are computed ONCE.
 * Training set = originals + `copies` augmented variants of every *training* image (augmenting after
 * the split so augmented copies can never leak into validation). Validation = originals only.
 */
export async function buildFeatureData(
  train: PreparedSample[], val: PreparedSample[], extractor: FeatureExtractor, spec: PreprocessingSpec,
  aug: AugmentationConfig, copies: number, seed: number, caps: AugmentCaps, numClasses: number,
  onProgress?: (done: number, total: number) => void, isAborted?: () => boolean,
): Promise<TrainData> {
  const D = extractor.featureDim;
  const size = spec.inputSize;
  const rng = mulberry32(seed ^ 0x9e3779b9);
  const passes = aug.enabled ? Math.max(0, copies) : 0;
  const total = train.length * (1 + passes) + val.length;
  let done = 0;

  const embedAll = async (samples: PreparedSample[], augment: boolean): Promise<Float32Array> => {
    const out = new Float32Array(samples.length * D);
    for (let i = 0; i < samples.length; i += CHUNK) {
      if (isAborted?.()) break;
      const end = Math.min(samples.length, i + CHUNK);
      const emb = tf.tidy(() => {
        let x = pixelsToUnit(stack(samples, i, end, size), end - i, size);
        if (augment) x = augmentBatch(x, aug, rng, caps);
        return extractor.embed(unitToSpec(x, spec));
      });
      out.set((await emb.data()) as Float32Array, i * D);
      emb.dispose();
      done += end - i;
      onProgress?.(done, total);
      await new Promise<void>((r) => setTimeout(r, 0));
    }
    return out;
  };

  const featureParts: Float32Array[] = [await embedAll(train, false)];
  const labelParts: number[][] = [train.map((s) => s.classIndex)];
  for (let p = 0; p < passes; p++) {
    featureParts.push(await embedAll(train, true));
    labelParts.push(train.map((s) => s.classIndex));
  }
  const valFeat = val.length ? await embedAll(val, false) : null;

  const xTrainBuf = new Float32Array(featureParts.reduce((a, f) => a + f.length, 0));
  let off = 0;
  featureParts.forEach((f) => {
    xTrainBuf.set(f, off);
    off += f.length;
  });
  const yTrain = Int32Array.from(labelParts.flat());
  return {
    xTrain: tf.tensor2d(xTrainBuf, [yTrain.length, D]),
    yTrain,
    xVal: valFeat ? tf.tensor2d(valFeat, [val.length, D]) : null,
    yVal: valFeat ? Int32Array.from(val.map((s) => s.classIndex)) : null,
    numClasses,
  };
}

/** From-scratch data: images stay on the device in unit range; augmentation is fresh for every batch. */
export function buildImageData(train: PreparedSample[], val: PreparedSample[], spec: PreprocessingSpec, config: TrainingConfig, caps: AugmentCaps, numClasses: number): TrainData {
  const size = spec.inputSize;
  const rng = mulberry32(config.seed ^ 0x85ebca6b);
  const xTrain = pixelsToUnit(stack(train, 0, train.length, size), train.length, size);
  const xVal = val.length ? unitToSpec(pixelsToUnit(stack(val, 0, val.length, size), val.length, size), spec) : null;
  return {
    xTrain,
    yTrain: Int32Array.from(train.map((s) => s.classIndex)),
    xVal,
    yVal: val.length ? Int32Array.from(val.map((s) => s.classIndex)) : null,
    numClasses,
    batchAugment: (x) => unitToSpec(augmentBatch(x as tf.Tensor4D, config.augmentation, rng, caps), spec),
  };
}
