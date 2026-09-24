import * as tf from '@tensorflow/tfjs';
import type { ModelArtifactsData, ModelMetadata, Prediction, PredictionResult } from '@/types/ml';
import type { FeatureExtractor } from '../models/backbone';
import { pixelsToTensor } from '../preprocessing/spec';

export const toModelArtifacts = (a: ModelArtifactsData): tf.io.ModelArtifacts => ({
  modelTopology: a.modelTopology as object,
  weightSpecs: a.weightSpecs as tf.io.WeightsManifestEntry[],
  weightData: a.weightData,
});

/**
 * Single inference path used by webcam, image tests and evaluation:
 * bytes -> (same preprocessing as training) -> [backbone] -> model -> probabilities.
 */
export class Predictor {
  constructor(
    readonly model: tf.LayersModel,
    readonly metadata: ModelMetadata,
    private readonly extractor: FeatureExtractor | null,
    private readonly ownsModel = true,
  ) {}

  static async fromArtifacts(artifacts: ModelArtifactsData, metadata: ModelMetadata, extractor: FeatureExtractor | null): Promise<Predictor> {
    const model = await tf.loadLayersModel(tf.io.fromMemory(toModelArtifacts(artifacts)));
    return new Predictor(model, metadata, extractor);
  }

  private classes() {
    return this.metadata.classes;
  }

  /** Probabilities for a batch of preprocessed-size RGB byte arrays (n * size*size*3). */
  predictProbs(pixels: Uint8Array, count: number): Float32Array {
    const spec = this.metadata.preprocessing;
    const probs = tf.tidy(() => {
      const x = pixelsToTensor(pixels, count, spec);
      const input = this.extractor ? this.extractor.embed(x) : x;
      return this.model.predict(input) as tf.Tensor;
    });
    const data = probs.dataSync() as Float32Array;
    probs.dispose();
    return data;
  }

  async predictPixels(pixels: Uint8Array, preprocessMs = 0): Promise<PredictionResult> {
    const t0 = performance.now();
    const spec = this.metadata.preprocessing;
    const probs = tf.tidy(() => {
      const x = pixelsToTensor(pixels, 1, spec);
      const input = this.extractor ? this.extractor.embed(x) : x;
      return this.model.predict(input) as tf.Tensor;
    });
    const data = (await probs.data()) as Float32Array; // includes GPU sync -> honest latency
    probs.dispose();
    const predictions: Prediction[] = this.classes().map((c, i) => ({ classId: c.id, className: c.name, probability: data[i] as number }));
    predictions.sort((a, b) => b.probability - a.probability);
    return { predictions, latencyMs: performance.now() - t0, preprocessMs };
  }

  dispose(): void {
    if (this.ownsModel) this.model.dispose();
  }
}
