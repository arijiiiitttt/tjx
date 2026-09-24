import { z } from 'zod';
import type { ModelArtifactsData, ModelMetadata } from '@/types/ml';

export const LIMITS = {
  modelJsonBytes: 20 * 1024 * 1024,
  weightBytes: 384 * 1024 * 1024,
  metadataBytes: 1024 * 1024,
  maxClasses: 500,
  zipEntries: 8,
} as const;

/** Layer types allowed in imported topologies. Anything else (custom / Lambda / unknown) is rejected. */
// Standard, side-effect-free Keras/TF.js layers. Deliberately excludes anything that can run
// arbitrary code (Lambda, custom layers) or reach outside the graph (I/O ops) — that is the actual
// security boundary, not "layers this app happens to generate itself". Widened against a real
// third-party export (see tests/fixtures/real-external-model.json) rather than only self-generated
// models, which is how Conv2DTranspose/Concatenate/DepthwiseConv2D/Add were found to be missing.
export const ALLOWED_LAYERS = new Set([
  'InputLayer', 'Conv2D', 'Conv2DTranspose', 'DepthwiseConv2D', 'SeparableConv2D',
  'MaxPooling2D', 'AveragePooling2D', 'GlobalAveragePooling2D', 'GlobalMaxPooling2D',
  'UpSampling2D', 'ZeroPadding2D', 'Cropping2D',
  'Flatten', 'Dense', 'Dropout', 'Activation', 'Softmax', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU',
  'Reshape', 'Permute', 'BatchNormalization', 'LayerNormalization',
  'Add', 'Subtract', 'Multiply', 'Average', 'Maximum', 'Minimum', 'Concatenate',
]);

const norm = z.object({ min: z.number().finite(), max: z.number().finite() });
export const metadataSchema = z.object({
  schemaVersion: z.literal(1),
  modelType: z.enum(['transfer-mobilenetv2', 'scratch-cnn']),
  classes: z.array(z.object({ id: z.string().min(1).max(64), name: z.string().min(1).max(80) })).min(2).max(LIMITS.maxClasses),
  inputShape: z.tuple([z.number().int().min(16).max(1024), z.number().int().min(16).max(1024), z.literal(3)]),
  modelInputShape: z.array(z.number().int().positive().max(1_000_000)).min(1).max(4),
  preprocessing: z.object({
    version: z.literal(1),
    inputSize: z.number().int().min(16).max(1024),
    resize: z.enum(['center-crop', 'pad', 'stretch']),
    colorSpace: z.literal('rgb'),
    normalization: norm,
  }),
  backbone: z.object({
    id: z.enum(['mobilenet-v2-100', 'mobilenet-v2-075', 'mobilenet-v2-050']),
    url: z.string().url().max(300),
    embeddingNode: z.string().max(200),
    featureDim: z.number().int().positive().max(100_000),
  }).nullable(),
  outputActivation: z.literal('softmax'),
  trainingConfig: z.record(z.string(), z.unknown()),
  finalMetrics: z.object({
    loss: z.number().nullable(), accuracy: z.number().nullable(), valLoss: z.number().nullable(),
    valAccuracy: z.number().nullable(), macroF1: z.number().nullable(),
  }),
  createdAt: z.string().max(64),
  appVersion: z.string().max(32),
});

export class ModelValidationError extends Error {
  constructor(message: string, readonly hint: string) {
    super(message);
    this.name = 'ModelValidationError';
  }
}

/**
 * Structural check that `topology` is a Keras/TF.js LayersModel graph, independent of the
 * top-level `format` string. Real-world exports (see tests/fixtures/real-external-model.json,
 * from a genuine external repository) sometimes omit `format: "layers-model"` even though the
 * topology itself is unambiguously a Keras model — requiring the literal string rejected valid
 * files. Structure, not a label, is what should be trusted here.
 */
export function looksLikeLayersTopology(topology: unknown): boolean {
  try {
    return extractLayers(topology).length > 0;
  } catch {
    return false;
  }
}

export function parseMetadata(json: unknown): ModelMetadata {
  const r = metadataSchema.safeParse(json);
  if (!r.success) {
    const first = r.error.issues[0];
    throw new ModelValidationError(
      `metadata.json is invalid at "${first?.path.join('.') ?? ''}": ${first?.message ?? 'unknown error'}.`,
      'Export the model again from this app, or make sure metadata.json was not edited.',
    );
  }
  return r.data as unknown as ModelMetadata;
}

interface LayerJson { class_name?: unknown; config?: Record<string, unknown> }

export function extractLayers(topology: unknown): LayerJson[] {
  // TF.js emits { class_name, config: { layers } }; Keras-converted models wrap this in { model_config }.
  const t = topology as { config?: unknown; model_config?: { config?: unknown } } | null;
  const cfg = t?.model_config?.config ?? t?.config;
  const layers = Array.isArray(cfg) ? cfg : (cfg as { layers?: unknown } | undefined)?.layers;
  if (!Array.isArray(layers)) throw new ModelValidationError('model.json has an unrecognised topology.', 'Only TF.js layers-format models (Sequential/functional) are supported.');
  return layers as LayerJson[];
}

export function specBytes(specs: unknown[]): number {
  return specs.reduce<number>((total, s) => {
    const spec = s as { shape?: number[]; dtype?: string; quantization?: { dtype?: string } };
    const count = (spec.shape ?? []).reduce((a, b) => a * b, 1);
    const per = spec.quantization ? (spec.quantization.dtype === 'uint8' ? 1 : 2) : spec.dtype === 'int32' || spec.dtype === 'float32' || !spec.dtype ? 4 : 1;
    return total + count * per;
  }, 0);
}

/** Structural checks on untrusted artifacts. Nothing here executes model code. */
/**
 * Layer-safety and weight-consistency checks that apply to ANY imported LayersModel — a full
 * classifier or a bare feature-extractor "backbone" someone supplies their own weights for.
 * Independent of ModelMetadata, which a bare backbone does not have.
 */
export function validateLayersArtifacts(artifacts: ModelArtifactsData): LayerJson[] {
  const layers = extractLayers(artifacts.modelTopology);
  for (const l of layers) {
    const name = String(l.class_name);
    if (!ALLOWED_LAYERS.has(name)) {
      throw new ModelValidationError(`Layer type "${name}" is not allowed.`, 'For safety only standard layers (Conv2D, Dense, Dropout, …) can be imported. Custom/Lambda layers are rejected.');
    }
  }
  if (artifacts.weightData.byteLength > LIMITS.weightBytes) {
    throw new ModelValidationError(`Weights are ${(artifacts.weightData.byteLength / 1e6).toFixed(0)} MB, above the ${LIMITS.weightBytes / 1e6} MB limit.`, 'Import a smaller model.');
  }
  const expected = specBytes(artifacts.weightSpecs);
  if (expected !== artifacts.weightData.byteLength) {
    throw new ModelValidationError(`weights.bin has ${artifacts.weightData.byteLength} bytes but model.json declares ${expected}.`, 'The weights file does not belong to this model.json (or is truncated). Re-download both files together.');
  }
  return layers;
}

export function validateArtifacts(artifacts: ModelArtifactsData, metadata: ModelMetadata): void {
  const layers = validateLayersArtifacts(artifacts);
  const first = layers[0]?.config as { batch_input_shape?: (number | null)[] } | undefined;
  const inShape = first?.batch_input_shape?.slice(1);
  if (inShape && inShape.join(',') !== metadata.modelInputShape.join(',')) {
    throw new ModelValidationError(`Model input shape [${inShape.join(', ')}] does not match metadata [${metadata.modelInputShape.join(', ')}].`, 'metadata.json belongs to a different model.');
  }
  const units = (layers.filter((l) => l.class_name === 'Dense').pop()?.config as { units?: number } | undefined)?.units;
  if (units !== undefined && units !== metadata.classes.length) {
    throw new ModelValidationError(`Model outputs ${units} classes but metadata lists ${metadata.classes.length}.`, 'metadata.json belongs to a different model.');
  }
  if (metadata.modelType === 'transfer-mobilenetv2' && (!metadata.backbone || metadata.backbone.featureDim !== metadata.modelInputShape[0])) {
    throw new ModelValidationError('A transfer-learning model must reference a backbone whose feature size matches the head input.', 'metadata.json is inconsistent.');
  }
  if (metadata.modelType === 'scratch-cnn' && metadata.modelInputShape.join(',') !== metadata.inputShape.join(',')) {
    throw new ModelValidationError('A from-scratch model must take the image tensor directly.', 'metadata.json is inconsistent.');
  }
}
