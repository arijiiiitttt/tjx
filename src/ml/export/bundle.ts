import { unzipSync, zipSync } from 'fflate';
import type { ModelArtifactsData, ModelMetadata } from '@/types/ml';
import { LIMITS, ModelValidationError, looksLikeLayersTopology, parseMetadata, validateArtifacts } from './validate';

const enc = new TextEncoder();
const dec = new TextDecoder();

export interface BundleFiles {
  'model.json': Uint8Array;
  'weights.bin': Uint8Array;
  'metadata.json': Uint8Array;
}

/** TF.js layers-format model.json + weights.bin (loadable by stock tf.loadLayersModel) plus metadata.json. */
export function buildBundleFiles(artifacts: ModelArtifactsData, metadata: ModelMetadata): BundleFiles {
  const modelJson = {
    format: 'layers-model',
    generatedBy: `browser-ml-studio ${metadata.appVersion}`,
    convertedBy: null,
    modelTopology: artifacts.modelTopology,
    weightsManifest: [{ paths: ['weights.bin'], weights: artifacts.weightSpecs }],
  };
  return {
    'model.json': enc.encode(JSON.stringify(modelJson)),
    'weights.bin': new Uint8Array(artifacts.weightData.slice(0)),
    'metadata.json': enc.encode(JSON.stringify(metadata, null, 2)),
  };
}

export const zipBundle = (files: BundleFiles): Uint8Array => zipSync({ ...files }, { level: 0 });

const ALLOWED_NAMES = new Set(['model.json', 'weights.bin', 'metadata.json']);

export interface NamedBytes {
  name: string;
  bytes: Uint8Array;
}

/** Accepts either loose files or one .zip. Files are matched by name; everything is size-checked first. */
export function readBundle(inputs: NamedBytes[]): { artifacts: ModelArtifactsData; metadata: ModelMetadata } {
  let files = new Map<string, Uint8Array>();
  const zip = inputs.find((i) => i.name.toLowerCase().endsWith('.zip'));
  if (zip) {
    if (zip.bytes.length > LIMITS.weightBytes + LIMITS.modelJsonBytes + LIMITS.metadataBytes) throw new ModelValidationError('The zip file is too large.', 'Import a smaller model.');
    let entries: Record<string, Uint8Array>;
    try {
      entries = unzipSync(zip.bytes, { filter: (f) => ALLOWED_NAMES.has(f.name.split('/').pop() ?? '') && f.originalSize <= LIMITS.weightBytes });
    } catch {
      throw new ModelValidationError('The zip file could not be read.', 'The archive is corrupted; export the model again.');
    }
    for (const [k, v] of Object.entries(entries)) files.set(k.split('/').pop() as string, v);
  } else {
    files = new Map(inputs.map((i) => [i.name.split('/').pop() as string, i.bytes]));
  }
  const modelBytes = files.get('model.json');
  const weights = files.get('weights.bin') ?? [...files.entries()].find(([n]) => n.endsWith('.bin'))?.[1];
  const metaBytes = files.get('metadata.json');
  if (!modelBytes) throw new ModelValidationError('model.json is missing.', 'Select model.json, weights.bin and metadata.json together (or the exported .zip).');
  if (!weights) throw new ModelValidationError('weights.bin is missing.', 'Select model.json, weights.bin and metadata.json together (or the exported .zip).');
  if (!metaBytes) throw new ModelValidationError('metadata.json is missing.', 'Without metadata the classes and preprocessing are unknown, so inference would be unreliable. Import files exported from this app.');
  if (modelBytes.length > LIMITS.modelJsonBytes || metaBytes.length > LIMITS.metadataBytes) throw new ModelValidationError('A JSON file exceeds the size limit.', 'Import a smaller model.');

  let modelJson: { format?: unknown; modelTopology?: unknown; weightsManifest?: { weights?: unknown[] }[] };
  let metaJson: unknown;
  try {
    modelJson = JSON.parse(dec.decode(modelBytes));
    metaJson = JSON.parse(dec.decode(metaBytes));
  } catch {
    throw new ModelValidationError('A JSON file could not be parsed.', 'The file is corrupted or not JSON.');
  }
  // Real-world tfjs_converter output sometimes omits the "format" field even for a genuine
  // Keras LayersModel (observed in third-party exports); trust the topology's structure instead
  // of requiring the literal string, while still rejecting anything that isn't actually a layers graph.
  if (!modelJson.modelTopology || !Array.isArray(modelJson.weightsManifest) || (modelJson.format && modelJson.format !== 'layers-model') || !looksLikeLayersTopology(modelJson.modelTopology)) {
    throw new ModelValidationError('model.json is not a recognisable TF.js layers-model.', 'Only TF.js layers-format models (Sequential/functional Keras graphs) are supported — not TF.js graph-models (frozen SavedModel/TFHub format).');
  }
  const weightSpecs = modelJson.weightsManifest.flatMap((g) => g.weights ?? []);
  const metadata = parseMetadata(metaJson);
  const artifacts: ModelArtifactsData = { modelTopology: modelJson.modelTopology, weightSpecs, weightData: weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength) as ArrayBuffer };
  validateArtifacts(artifacts, metadata);
  return { artifacts, metadata };
}
