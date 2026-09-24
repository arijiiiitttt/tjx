import type { DatasetClass } from './domain';

export type BackendName = 'webgpu' | 'webgl' | 'wasm' | 'cpu';
export type BackendPreference = 'auto' | BackendName;
export type TrainingMode = 'transfer' | 'scratch';
export type BackboneId = 'mobilenet-v2-100' | 'mobilenet-v2-075' | 'mobilenet-v2-050';

export interface Normalization {
  min: number;
  max: number;
}

/** Single source of truth for how pixels become tensors. Stored with every model. */
export interface PreprocessingSpec {
  version: 1;
  inputSize: number;
  resize: 'center-crop' | 'pad' | 'stretch';
  colorSpace: 'rgb';
  normalization: Normalization;
}

export interface AugmentationConfig {
  enabled: boolean;
  hFlip: boolean;
  /** Max rotation in degrees (0 = off). */
  rotationDeg: number;
  /** Max zoom deviation, e.g. 0.15 => scale in [0.85, 1.15] (0 = off). */
  zoom: number;
  /** Max translation as a fraction of the image size (0 = off). */
  translate: number;
  brightness: number;
  contrast: number;
}

export interface TrainingConfig {
  mode: TrainingMode;
  backboneId: BackboneId;
  /** When set, use this imported custom backbone instead of backboneId. */
  customBackboneId: string | null;
  epochs: number;
  batchSize: number;
  learningRate: number;
  validationSplit: number;
  seed: number;
  augmentation: AugmentationConfig;
  /** Transfer mode: number of augmented feature copies per training image. */
  augmentedCopies: number;
  /** 0 disables early stopping. */
  earlyStoppingPatience: number;
  classWeighting: 'none' | 'balanced';
  scratchInputSize: number;
  scratchFilters: number[];
  hiddenUnits: number;
  dropout: number;
  /** Scratch mode: emit gradient/weight statistics every N steps (0 = off). */
  traceEvery: number;
}

export interface BatchMetrics {
  epoch: number;
  batch: number;
  batches: number;
  loss: number;
  accuracy: number;
  batchMs: number;
  samplesPerSec: number;
}

export interface EpochMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  valLoss: number | null;
  valAccuracy: number | null;
  epochMs: number;
  samplesPerSec: number;
}

export interface LayerTrace {
  name: string;
  gradNorm: number;
  weightNorm: number;
  updateNorm: number;
}
export interface MechanicsTrace {
  step: number;
  epoch: number;
  loss: number;
  layers: LayerTrace[];
}

export interface ClassMetrics {
  classId: string;
  className: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface MisclassifiedSample {
  sampleId: string;
  trueClassId: string;
  predictedClassId: string;
  probability: number;
  trueProbability: number;
}

export interface EvaluationReport {
  sampleCount: number;
  accuracy: number;
  macroF1: number;
  weightedF1: number;
  perClass: ClassMetrics[];
  /** confusion[actual][predicted] */
  confusion: number[][];
  classIds: string[];
  classNames: string[];
  misclassified: MisclassifiedSample[];
}

export interface Prediction {
  classId: string;
  className: string;
  probability: number;
}

export interface PredictionResult {
  predictions: Prediction[];
  latencyMs: number;
  preprocessMs: number;
}

export type ModelType = 'transfer-mobilenetv2' | 'scratch-cnn';

export interface BackboneRef {
  id: BackboneId;
  url: string;
  embeddingNode: string;
  featureDim: number;
}

export interface ModelMetadata {
  schemaVersion: 1;
  modelType: ModelType;
  classes: Pick<DatasetClass, 'id' | 'name'>[];
  /** Shape of one *image* fed to preprocessing (H, W, C). */
  inputShape: [number, number, number];
  /** Shape fed to the exported TF.js layers model (batch dimension omitted). */
  modelInputShape: number[];
  preprocessing: PreprocessingSpec;
  backbone: BackboneRef | null;
  customBackbone: { id: string; name: string; featureDim: number } | null;
  outputActivation: 'softmax';
  trainingConfig: TrainingConfig;
  finalMetrics: {
    loss: number | null;
    accuracy: number | null;
    valLoss: number | null;
    valAccuracy: number | null;
    macroF1: number | null;
  };
  createdAt: string;
  appVersion: string;
}

export interface ModelArtifactsData {
  modelTopology: unknown;
  weightSpecs: unknown[];
  weightData: ArrayBuffer;
}

export interface ModelRecord {
  id: string;
  projectId: string | null;
  name: string;
  createdAt: number;
  metadata: ModelMetadata;
  experimentId: string | null;
  sizeBytes: number;
  imported: boolean;
}

export interface PerfTimings {
  decodeMs?: number;
  backboneLoadMs?: number;
  featureExtractionMs?: number;
  splitMs?: number;
  trainMs?: number;
  evaluateMs?: number;
  totalMs?: number;
}

export interface ExperimentRecord {
  id: string;
  number: number;
  projectId: string;
  createdAt: number;
  datasetHash: string;
  sampleCount: number;
  classNames: string[];
  modelType: ModelType;
  architecture: string;
  config: TrainingConfig;
  backend: BackendName;
  epochsRun: number;
  history: EpochMetrics[];
  finalMetrics: ModelMetadata['finalMetrics'];
  durationMs: number;
  paramCount: number;
  modelBytes: number;
  timings: PerfTimings;
  modelId: string | null;
  trainIds: string[];
  valIds: string[];
  evaluation: EvaluationReport | null;
  stoppedEarly: boolean;
  aborted: boolean;
}

export interface CustomBackboneRecord {
  id: string;
  name: string;
  createdAt: number;
  featureDim: number;
  inputSize: number;
  sizeBytes: number;
}

export interface BackendAttempt {
  backend: BackendName;
  ok: boolean;
  error?: string;
  probeMs?: number;
}

export interface RuntimeReport {
  activeBackend: BackendName;
  attempts: BackendAttempt[];
  affineAugmentation: boolean;
  /** false when the backend's batched Transform kernel returns wrong data (augmentation then runs per sample). */
  affineBatch: boolean;
  host: 'worker' | 'inline';
  gpu: { vendor?: string; architecture?: string; renderer?: string } | null;
}

export interface MemoryReport {
  numTensors: number;
  numBytes: number;
  numBytesInGPU: number | null;
  unreliable: boolean;
}

export interface GradientCheckReport {
  backend: BackendName;
  passed: boolean;
  maxRelError: number;
  checked: number;
  tolerance: number;
  samples: { variable: string; analytic: number; numeric: number }[];
}
