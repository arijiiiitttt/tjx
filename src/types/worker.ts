import type {
  BackboneId, BackendName, BackendPreference, EpochMetrics, EvaluationReport, GradientCheckReport,
  BatchMetrics, MechanicsTrace, MemoryReport, ModelArtifactsData, ModelMetadata, PerfTimings,
  PredictionResult, RuntimeReport, TrainingConfig,
} from './ml';

export interface CustomBackboneInput {
  id: string;
  name: string;
  artifacts: ModelArtifactsData;
}

export interface RawSample {
  id: string;
  classId: string;
  groupId: string;
  seq: number;
  blob: Blob;
}

export interface TrainRequest {
  classes: { id: string; name: string }[];
  samples: RawSample[];
  config: TrainingConfig;
  /** Required when config.customBackboneId is set. */
  customBackbone: CustomBackboneInput | null;
}

export interface TrainResponse {
  metadata: ModelMetadata;
  artifacts: ModelArtifactsData;
  evaluation: EvaluationReport | null;
  history: EpochMetrics[];
  trainIds: string[];
  valIds: string[];
  purgedIds: string[];
  failedSampleIds: string[];
  durationMs: number;
  paramCount: number;
  modelBytes: number;
  timings: PerfTimings;
  backend: BackendName;
  stoppedEarly: boolean;
  aborted: boolean;
  warnings: string[];
}

export type TrainPhase = 'decoding' | 'backbone' | 'features' | 'training' | 'evaluating' | 'finalizing';

export interface MlEvents {
  'train:phase': { phase: TrainPhase; done?: number; total?: number; message?: string };
  'train:epochStart': { epoch: number; epochs: number };
  'train:batch': BatchMetrics;
  'train:epoch': EpochMetrics;
  'train:validation': { epoch: number; valLoss: number; valAccuracy: number };
  'train:mechanics': MechanicsTrace;
  'backbone:progress': { id: BackboneId; fraction: number };
}

/** The API exposed by the ML runtime host (worker or inline). */
export interface MlApi {
  init(pref: BackendPreference): Promise<RuntimeReport>;
  memory(): Promise<MemoryReport>;
  train(req: TrainRequest): Promise<TrainResponse>;
  loadModel(artifacts: ModelArtifactsData, metadata: ModelMetadata, customBackbone?: CustomBackboneInput | null): Promise<{ loadMs: number }>;
  unloadModel(): Promise<void>;
  predictBitmap(bitmap: ImageBitmap): Promise<PredictionResult>;
  evaluate(samples: RawSample[]): Promise<EvaluationReport>;
  gradientCheck(): Promise<GradientCheckReport>;
  prefetchBackbone(id: BackboneId): Promise<{ cached: boolean; ms: number }>;
}

export interface MlHost extends MlApi {
  readonly kind: 'worker' | 'inline';
  abortTraining(): void;
  on<K extends keyof MlEvents>(name: K, cb: (payload: MlEvents[K]) => void): () => void;
  terminate(): void;
}
