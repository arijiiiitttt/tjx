import type { PreprocessingSpec } from '@/types/ml';

export const MOBILENET_SPEC: PreprocessingSpec = {
  version: 1,
  inputSize: 224,
  resize: 'center-crop',
  colorSpace: 'rgb',
  // TF Hub MobileNetV2 (imagenet/classification/2) expects inputs in [0, 1].
  normalization: { min: 0, max: 1 },
};

export function scratchSpec(inputSize: number): PreprocessingSpec {
  return { version: 1, inputSize, resize: 'center-crop', colorSpace: 'rgb', normalization: { min: 0, max: 1 } };
}

