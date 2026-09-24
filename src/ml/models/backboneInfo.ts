import type { BackboneId, BackboneRef } from '@/types/ml';

export interface BackboneInfo {
  id: BackboneId;
  label: string;
  url: string;
  embeddingNode: string;
  featureDim: number;
  approxParams: string;
  note: string;
}

const NODE = 'module_apply_default/MobilenetV2/Logits/AvgPool';

export const BACKBONES: Record<BackboneId, BackboneInfo> = {
  'mobilenet-v2-100': {
    id: 'mobilenet-v2-100', label: 'MobileNetV2 ×1.0', url: 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2',
    embeddingNode: NODE, featureDim: 1280, approxParams: '~3.5M params (~14 MB)', note: 'Best accuracy of the three; default.',
  },
  'mobilenet-v2-075': {
    id: 'mobilenet-v2-075', label: 'MobileNetV2 ×0.75', url: 'https://tfhub.dev/google/imagenet/mobilenet_v2_075_224/classification/2',
    embeddingNode: NODE, featureDim: 1280, approxParams: '~2.6M params (~10 MB)', note: 'Balanced.',
  },
  'mobilenet-v2-050': {
    id: 'mobilenet-v2-050', label: 'MobileNetV2 ×0.5', url: 'https://tfhub.dev/google/imagenet/mobilenet_v2_050_224/classification/2',
    embeddingNode: NODE, featureDim: 1280, approxParams: '~2.0M params (~8 MB)', note: 'Fastest; for weak GPUs and phones.',
  },
};

export const backboneRef = (id: BackboneId): BackboneRef => {
  const b = BACKBONES[id];
  return { id, url: b.url, embeddingNode: b.embeddingNode, featureDim: b.featureDim };
};

