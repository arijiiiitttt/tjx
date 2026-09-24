import type { ImageSample, Project } from '@/types/domain';
import type { CustomBackboneRecord, ExperimentRecord, ModelArtifactsData, ModelRecord } from '@/types/ml';

export interface SampleBlobs {
  /** The untouched original bytes. */
  blob: Blob;
  /** ~128 px JPEG for fast grids. */
  thumb: Blob;
}

export interface StoredModel {
  record: ModelRecord;
  artifacts: ModelArtifactsData;
}

export interface StorageEstimate {
  usage: number;
  quota: number;
  persisted: boolean | null;
}

/**
 * Persistence boundary. UI code depends on this interface only; IndexedDB is one implementation
 * (in-memory is used as a visible fallback when IndexedDB is unavailable, and in tests).
 */
export interface AppStorage {
  readonly kind: 'indexeddb' | 'memory';
  projects: {
    list(): Promise<Project[]>;
    get(id: string): Promise<Project | undefined>;
    put(project: Project): Promise<void>;
    delete(id: string): Promise<void>;
  };
  samples: {
    list(projectId: string): Promise<ImageSample[]>;
    put(sample: ImageSample, blobs: SampleBlobs): Promise<void>;
    getBlob(id: string): Promise<Blob | undefined>;
    getThumb(id: string): Promise<Blob | undefined>;
    delete(ids: string[]): Promise<void>;
  };
  models: {
    list(projectId?: string): Promise<ModelRecord[]>;
    put(record: ModelRecord, artifacts: ModelArtifactsData): Promise<void>;
    get(id: string): Promise<StoredModel | undefined>;
    delete(id: string): Promise<void>;
  };
  customBackbones: {
    list(): Promise<CustomBackboneRecord[]>;
    put(record: CustomBackboneRecord, artifacts: ModelArtifactsData): Promise<void>;
    get(id: string): Promise<{ record: CustomBackboneRecord; artifacts: ModelArtifactsData } | undefined>;
    delete(id: string): Promise<void>;
  };
  experiments: {
    list(projectId: string): Promise<ExperimentRecord[]>;
    put(experiment: ExperimentRecord): Promise<void>;
    delete(id: string): Promise<void>;
  };
  settings: {
    get<T>(key: string): Promise<T | undefined>;
    set<T>(key: string, value: T): Promise<void>;
  };
  estimate(): Promise<StorageEstimate | null>;
  requestPersistence(): Promise<boolean>;
  clearAll(): Promise<void>;
}

export class StorageError extends Error {
  constructor(message: string, readonly hint: string, readonly original?: unknown) {
    super(message);
    this.name = 'StorageError';
  }
}
