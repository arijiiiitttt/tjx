import * as tf from '@tensorflow/tfjs';

export async function useCpu(): Promise<void> {
  await tf.setBackend('cpu');
  await tf.ready();
}

/** Deterministic synthetic RGB images: class 0 = horizontal stripes, class 1 = vertical stripes (+ noise, phase jitter). */
export function stripeImage(size: number, cls: number, rng: () => number): Uint8Array {
  const px = new Uint8Array(size * size * 3);
  const period = 6 + Math.floor(rng() * 3);
  const phase = Math.floor(rng() * period);
  const tint = [rng() * 40, rng() * 40, rng() * 40];
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const t = cls === 0 ? y : x;
      const on = Math.floor((t + phase) / (period / 2)) % 2 === 0;
      const base = on ? 210 : 45;
      for (let c = 0; c < 3; c++) {
        const v = base + (tint[c] as number) + (rng() - 0.5) * 30;
        px[(y * size + x) * 3 + c] = Math.max(0, Math.min(255, v));
      }
    }
  }
  return px;
}

/** Colour-dominant images: class 0 reddish, class 1 bluish. Used with a fake frozen backbone. */
export function colourImage(size: number, cls: number, rng: () => number): Uint8Array {
  const px = new Uint8Array(size * size * 3);
  for (let i = 0; i < size * size; i++) {
    const hi = 170 + rng() * 60;
    const lo = 30 + rng() * 60;
    px[i * 3] = cls === 0 ? hi : lo;
    px[i * 3 + 1] = 60 + rng() * 40;
    px[i * 3 + 2] = cls === 0 ? lo : hi;
  }
  return px;
}

/** Class 0: bright square in the LEFT half; class 1: RIGHT half. (Horizontal flip would swap the labels.) */
export function blobImage(size: number, cls: number, rng: () => number): Uint8Array {
  const px = new Uint8Array(size * size * 3);
  const bw = Math.floor(size / 3);
  const half = Math.floor(size / 2);
  const x0 = (cls === 0 ? 0 : half) + Math.floor(rng() * (half - bw));
  const y0 = Math.floor(rng() * (size - bw));
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const inside = x >= x0 && x < x0 + bw && y >= y0 && y < y0 + bw;
      for (let c = 0; c < 3; c++) px[(y * size + x) * 3 + c] = Math.max(0, Math.min(255, (inside ? 215 : 60) + (rng() - 0.5) * 40));
    }
  }
  return px;
}
