import * as tf from '@tensorflow/tfjs';
import type { BackboneId, PreprocessingSpec } from '@/types/ml';
import { MOBILENET_SPEC } from '../preprocessing/spec';

import { BACKBONES } from './backboneInfo';
export { BACKBONES, backboneRef, type BackboneInfo } from './backboneInfo';
import { validateLayersArtifacts } from '../export/validate';
import type { ModelArtifactsData } from '@/types/ml';

/** A frozen feature extractor: normalised image batch -> [batch, featureDim] embeddings. */
export interface FeatureExtractor {
  id: string;
  spec: PreprocessingSpec;
  featureDim: number;
  embed(x: tf.Tensor4D): tf.Tensor2D;
  dispose(): void;
}

const cacheUrl = (id: BackboneId) => `indexeddb://backbone-${id}`;

export interface BackboneLoadResult {
  extractor: FeatureExtractor;
  cached: boolean;
  cacheWriteFailed: boolean;
}

/**
 * Loads a backbone from the IndexedDB cache when present; otherwise downloads it once from TF Hub
 * and caches it, so later runs work offline.
 */
export async function loadBackbone(id: BackboneId, onProgress?: (fraction: number) => void): Promise<BackboneLoadResult> {
  const info = BACKBONES[id];
  let model: tf.GraphModel;
  let cached = false;
  let cacheWriteFailed = false;
  try {
    model = await tf.loadGraphModel(cacheUrl(id));
    cached = true;
  } catch {
    try {
      model = await tf.loadGraphModel(info.url, { fromTFHub: true, onProgress });
    } catch (e) {
      throw new Error(
        `Could not download ${info.label} from TF Hub (${e instanceof Error ? e.message : String(e)}). ` +
          'Transfer learning needs a one-time download of the backbone; check your connection, or use "From scratch" mode which needs no download.',
      );
    }
    try {
      await model.save(cacheUrl(id));
    } catch {
      cacheWriteFailed = true;
    }
  }
  const m = model;
  const extractor: FeatureExtractor = {
    id,
    spec: MOBILENET_SPEC,
    featureDim: info.featureDim,
    embed: (x) => tf.tidy(() => (m.execute(x, info.embeddingNode) as tf.Tensor).squeeze([1, 2]) as tf.Tensor2D),
    dispose: () => m.dispose(),
  };
  // Warm-up compiles shaders so the first real batch is not penalised.
  tf.tidy(() => extractor.embed(tf.zeros([1, 224, 224, 3])));
  return { extractor, cached, cacheWriteFailed };
}

/**
 * Loads a user-supplied LayersModel (model.json + weights.bin, no metadata.json needed) as a
 * frozen feature extractor. This is the escape hatch for restricted networks where the built-in
 * TF Hub backbones cannot be downloaded: the person fetches a TF.js layers-format model.json +
 * weights.bin from anywhere (their own conversion, a mirror, another machine), the same
 * layer/weight validation used for full model imports applies, and the model's own output is
 * used directly as the embedding (spatial 4D output is global-average-pooled to 2D automatically,
 * since that is what a classifier head expects and is a lossless no-op for an already-pooled output).
 */
export async function loadCustomBackbone(id: string, artifacts: ModelArtifactsData): Promise<FeatureExtractor> {
  validateLayersArtifacts(artifacts);
  const model = await tf.loadLayersModel(tf.io.fromMemory({
    modelTopology: artifacts.modelTopology as object,
    weightSpecs: artifacts.weightSpecs as tf.io.WeightsManifestEntry[],
    weightData: artifacts.weightData,
  }));
  const outShape = model.outputs[0]?.shape ?? [];
  if (outShape.length < 2 || outShape.length > 4) {
    model.dispose();
    throw new Error(`This model's output has ${outShape.length} dimensions; a feature-extractor backbone must output a 2D (already-pooled) or 4D (spatial) tensor.`);
  }
  const featureDim = outShape[outShape.length - 1] as number;
  if (!featureDim || featureDim < 2 || featureDim > 8192) {
    model.dispose();
    throw new Error(`This model's output feature size (${featureDim}) looks wrong for a backbone. Export the model up to its pooling layer, not the final classification layer.`);
  }
  const spatial = outShape.length === 4;
  const spec: PreprocessingSpec = { ...MOBILENET_SPEC, inputSize: (model.inputs[0]?.shape?.[1] as number) || 224 };
  return {
    id: `custom:${id}`,
    spec,
    featureDim,
    embed: (x) => tf.tidy(() => {
      const out = model.predict(x) as tf.Tensor;
      return (spatial ? out.mean([1, 2]) : out) as tf.Tensor2D;
    }),
    dispose: () => model.dispose(),
  };
}
