import type { AugmentationConfig } from '@/types/ml';

export const DEFAULT_AUGMENTATION: AugmentationConfig = {
  enabled: true,
  hFlip: true,
  rotationDeg: 12,
  zoom: 0.12,
  translate: 0.08,
  brightness: 0.12,
  contrast: 0.15,
};

export const NO_AUGMENTATION: AugmentationConfig = { ...DEFAULT_AUGMENTATION, enabled: false };

/**
 * Builds the 8-parameter projective transform tf.image.transform expects (output -> input mapping):
 * src = c + R(theta) * (p - c) / zoom + t
 */
export function affineParams(w: number, h: number, theta: number, zoom: number, tx: number, ty: number): number[] {
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  const a0 = cos / zoom;
  const a1 = -sin / zoom;
  const a2 = cx - (cos * cx - sin * cy) / zoom + tx;
  const b0 = sin / zoom;
  const b1 = cos / zoom;
  const b2 = cy - (sin * cx + cos * cy) / zoom + ty;
  return [a0, a1, a2, b0, b1, b2, 0, 0];
}

export const AUGMENTATION_GUIDE: { key: keyof AugmentationConfig; label: string; safeFor: string; riskFor: string }[] = [
  { key: 'hFlip', label: 'Horizontal flip', safeFor: 'objects, animals, scenes', riskFor: 'text, digits, left/right-hand or arrow-direction classes' },
  { key: 'rotationDeg', label: 'Rotation', safeFor: 'plants, food, textures, handheld objects', riskFor: 'orientation-defined classes (6 vs 9, up vs down, clock hands)' },
  { key: 'zoom', label: 'Zoom', safeFor: 'most objects', riskFor: 'small objects that may be cropped out of frame' },
  { key: 'translate', label: 'Translation', safeFor: 'most objects', riskFor: 'position-defined classes (object left vs right)' },
  { key: 'brightness', label: 'Brightness', safeFor: 'robustness to lighting', riskFor: 'classes defined by exposure (day vs night)' },
  { key: 'contrast', label: 'Contrast', safeFor: 'robustness to camera settings', riskFor: 'classes defined by contrast or haze' },
];
