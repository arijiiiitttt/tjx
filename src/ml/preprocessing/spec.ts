import * as tf from '@tensorflow/tfjs';
import type { PreprocessingSpec } from '@/types/ml';

export { MOBILENET_SPEC, scratchSpec } from './specs';

export const CHANNELS = 3;

/**
 * THE canonical pixel -> tensor path. Training, evaluation and inference all call these two
 * functions, so preprocessing cannot silently diverge.
 *
 *   uint8 RGB bytes  --pixelsToUnit-->  float [0,1]  --unitToSpec-->  model input range
 *
 * Augmentation (training only) operates between the two steps, in unit range.
 */
export function pixelsToUnit(pixels: Uint8Array | Uint8ClampedArray | Int32Array, count: number, size: number): tf.Tensor4D {
  return tf.tidy(() => tf.tensor4d(pixels as unknown as Int32Array, [count, size, size, CHANNELS], 'int32').toFloat().div(255));
}

export function unitToSpec(x: tf.Tensor4D, spec: PreprocessingSpec): tf.Tensor4D {
  const { min, max } = spec.normalization;
  if (min === 0 && max === 1) return x;
  return tf.tidy(() => x.mul(max - min).add(min) as tf.Tensor4D);
}

export function pixelsToTensor(pixels: Uint8Array, count: number, spec: PreprocessingSpec): tf.Tensor4D {
  return tf.tidy(() => unitToSpec(pixelsToUnit(pixels, count, spec.inputSize), spec));
}

export function specKey(spec: PreprocessingSpec): string {
  return `${spec.version}:${spec.inputSize}:${spec.resize}:${spec.colorSpace}:${spec.normalization.min}:${spec.normalization.max}`;
}
