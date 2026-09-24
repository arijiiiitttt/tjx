import * as tf from '@tensorflow/tfjs';
import type { AugmentationConfig } from '@/types/ml';

import { affineParams } from './config';
export { AUGMENTATION_GUIDE, DEFAULT_AUGMENTATION, NO_AUGMENTATION, affineParams } from './config';

export interface AugmentCaps {
  /** false when the active backend cannot run tf.image.transform; geometric ops are then skipped. */
  affine: boolean;
  /** false: the batched kernel is unreliable on this backend; transform one image at a time. Default true. */
  affineBatch?: boolean;
}

const uniform = (rng: () => number, amp: number) => (rng() * 2 - 1) * amp;

/**
 * On-device augmentation of a batch in unit range [0,1]. All random parameters are drawn from
 * the supplied seeded RNG (JS side), so a run is reproducible regardless of backend.
 * Returns a tensor owned by the caller's tidy scope; never call on validation data.
 */
export function augmentBatch(x: tf.Tensor4D, cfg: AugmentationConfig, rng: () => number, caps: AugmentCaps): tf.Tensor4D {
  if (!cfg.enabled) return x;
  return tf.tidy(() => {
    const [b, h, w] = x.shape as [number, number, number, number];
    let out = x;

    if (cfg.hFlip) {
      // Per-sample blend with a {0,1} mask (backend-agnostic; tf.where does not broadcast a 1-D mask over 4-D).
      const m = tf.tensor4d(Array.from({ length: b }, () => (rng() < 0.5 ? 1 : 0)), [b, 1, 1, 1]);
      out = tf.reverse(out, [2]).mul(m).add(out.mul(tf.sub(1, m))) as tf.Tensor4D;
    }

    const geometric = cfg.rotationDeg > 0 || cfg.zoom > 0 || cfg.translate > 0;
    if (geometric && caps.affine) {
      const params: number[] = [];
      for (let i = 0; i < b; i++) {
        const theta = (uniform(rng, cfg.rotationDeg) * Math.PI) / 180;
        const zoom = 1 + uniform(rng, cfg.zoom);
        params.push(...affineParams(w, h, theta, zoom, uniform(rng, cfg.translate) * w, uniform(rng, cfg.translate) * h));
      }
      if (caps.affineBatch === false) {
        const tiles: tf.Tensor4D[] = [];
        for (let i = 0; i < b; i++) {
          const one = out.slice([i, 0, 0, 0], [1, h, w, -1]) as tf.Tensor4D;
          tiles.push(tf.image.transform(one, tf.tensor2d(params.slice(i * 8, i * 8 + 8), [1, 8]), 'bilinear', 'reflect', 0) as tf.Tensor4D);
        }
        out = tf.concat(tiles, 0) as tf.Tensor4D;
      } else {
        out = tf.image.transform(out, tf.tensor2d(params, [b, 8]), 'bilinear', 'reflect', 0) as tf.Tensor4D;
      }
    }

    if (cfg.contrast > 0) {
      const c = tf.tensor4d(Array.from({ length: b }, () => 1 + uniform(rng, cfg.contrast)), [b, 1, 1, 1]);
      const mean = out.mean([1, 2, 3], true);
      out = out.sub(mean).mul(c).add(mean) as tf.Tensor4D;
    }
    if (cfg.brightness > 0) {
      const d = tf.tensor4d(Array.from({ length: b }, () => uniform(rng, cfg.brightness)), [b, 1, 1, 1]);
      out = out.add(d) as tf.Tensor4D;
    }
    return out.clipByValue(0, 1) as tf.Tensor4D;
  });
}

