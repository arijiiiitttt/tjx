import type { ImageSample, Project } from '@/types/domain';
import type { CustomBackboneRecord, ExperimentRecord, ModelArtifactsData, ModelRecord } from '@/types/ml';
import type { AppStorage, SampleBlobs } from './types';

export function createMemoryStorage(): AppStorage {
  const projects = new Map<string, Project>();
  const samples = new Map<string, ImageSample>();
  const blobs = new Map<string, SampleBlobs>();
  const models = new Map<string, { record: ModelRecord; artifacts: ModelArtifactsData }>();
  const backbones = new Map<string, { record: CustomBackboneRecord; artifacts: ModelArtifactsData }>();
  const experiments = new Map<string, ExperimentRecord>();
  const settings = new Map<string, unknown>();
  return {
    kind: 'memory',
    projects: {
      list: async () => [...projects.values()].sort((a, b) => b.updatedAt - a.updatedAt),
      get: async (id) => projects.get(id),
      put: async (p) => void projects.set(p.id, p),
      delete: async (id) => {
        projects.delete(id);
        for (const s of [...samples.values()]) if (s.projectId === id) { samples.delete(s.id); blobs.delete(s.id); }
        // custom backbones are cross-project (like built-in ones) and are not deleted with a project
        for (const [k, m] of models) if (m.record.projectId === id) models.delete(k);
        for (const [k, e] of experiments) if (e.projectId === id) experiments.delete(k);
      },
    },
    samples: {
      list: async (pid) => [...samples.values()].filter((s) => s.projectId === pid).sort((a, b) => a.createdAt - b.createdAt),
      put: async (s, b) => void (samples.set(s.id, s), blobs.set(s.id, b)),
      getBlob: async (id) => blobs.get(id)?.blob,
      getThumb: async (id) => blobs.get(id)?.thumb,
      delete: async (ids) => ids.forEach((id) => (samples.delete(id), blobs.delete(id))),
    },
    models: {
      list: async (pid) => [...models.values()].map((m) => m.record).filter((r) => !pid || r.projectId === pid).sort((a, b) => b.createdAt - a.createdAt),
      put: async (record, artifacts) => void models.set(record.id, { record, artifacts }),
      get: async (id) => models.get(id),
      delete: async (id) => void models.delete(id),
    },
    customBackbones: {
      list: async () => [...backbones.values()].map((b) => b.record).sort((a, b) => b.createdAt - a.createdAt),
      put: async (record, artifacts) => void backbones.set(record.id, { record, artifacts }),
      get: async (id) => backbones.get(id),
      delete: async (id) => void backbones.delete(id),
    },
    experiments: {
      list: async (pid) => [...experiments.values()].filter((e) => e.projectId === pid).sort((a, b) => b.number - a.number),
      put: async (e) => void experiments.set(e.id, e),
      delete: async (id) => void experiments.delete(id),
    },
    settings: {
      get: async <T>(k: string) => settings.get(k) as T | undefined,
      set: async (k, v) => void settings.set(k, v),
    },
    estimate: async () => null,
    requestPersistence: async () => false,
    clearAll: async () => {
      [projects, samples, blobs, models, backbones, experiments, settings].forEach((m) => m.clear());
    },
  };
}
