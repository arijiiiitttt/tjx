import { DEFAULT_AUGMENTATION } from '@/ml/augmentation/config';
import type { BackendPreference, TrainingConfig, TrainingMode } from '@/types/ml';

export interface AppSettings {
  backend: BackendPreference;
  seed: number;
  validationSplit: number;
  threshold: number;
  inferenceFps: number;
}

export const DEFAULT_SETTINGS: AppSettings = { backend: 'auto', seed: 42, validationSplit: 0.2, threshold: 0.7, inferenceFps: 10 };

export function defaultTrainingConfig(mode: TrainingMode, s: AppSettings): TrainingConfig {
  const common = {
    mode, backboneId: 'mobilenet-v2-100' as const, customBackboneId: null, validationSplit: s.validationSplit, seed: s.seed,
    augmentation: { ...DEFAULT_AUGMENTATION }, classWeighting: 'balanced' as const, dropout: 0.3, traceEvery: 10,
    scratchFilters: [16, 32, 64], scratchInputSize: 64,
  };
  return mode === 'transfer'
    ? { ...common, epochs: 30, batchSize: 32, learningRate: 0.001, augmentedCopies: 3, earlyStoppingPatience: 8, hiddenUnits: 128 }
    : { ...common, epochs: 40, batchSize: 16, learningRate: 0.002, augmentedCopies: 0, earlyStoppingPatience: 0, hiddenUnits: 64, dropout: 0.3 };
}

export { CLASS_COLORS } from './palette';
