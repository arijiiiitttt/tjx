import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { ImageSample, Project } from '@/types/domain';
import type { CustomBackboneRecord, ExperimentRecord, ModelArtifactsData, ModelRecord } from '@/types/ml';
import { StorageError, type AppStorage, type SampleBlobs } from './types';

interface Schema extends DBSchema {
  projects: { key: string; value: Project };
  samples: { key: string; value: ImageSample; indexes: { 'by-project': string } };
  blobs: { key: string; value: SampleBlobs };
  models: { key: string; value: ModelRecord; indexes: { 'by-project': string } };
  modelArtifacts: { key: string; value: ModelArtifactsData };
  experiments: { key: string; value: ExperimentRecord; indexes: { 'by-project': string } };
  customBackbones: { key: string; value: CustomBackboneRecord };
  customBackboneArtifacts: { key: string; value: ModelArtifactsData };
  settings: { key: string; value: unknown };
}

const DB_NAME = 'browser-ml-studio';
const VERSION = 1;

function describe(e: unknown): StorageError {
  const name = (e as { name?: string })?.name;
  if (name === 'QuotaExceededError') {
    return new StorageError('The browser storage quota is full.', 'Delete unused projects, models or images (Settings → Storage), or export models and free space.', e);
  }
  if (name === 'InvalidStateError' || name === 'SecurityError') {
    return new StorageError('IndexedDB is not available (private browsing or blocked by browser settings).', 'Allow site storage or leave private mode. The app can still run, but nothing will be saved.', e);
  }
  return new StorageError(`Storage operation failed: ${e instanceof Error ? e.message : String(e)}`, 'Reload the page. If it persists, clear this site\'s data.', e);
}

async function guard<T>(fn: () => Promise<T>): Promise<T> {
  try {
    return await fn();
  } catch (e) {
    throw e instanceof StorageError ? e : describe(e);
  }
}

export async function createIdbStorage(): Promise<AppStorage> {
  if (typeof indexedDB === 'undefined') throw new StorageError('IndexedDB is not supported in this browser.', 'Use a current version of Chrome, Edge, Firefox or Safari.');
  let db: IDBPDatabase<Schema>;
  try {
    db = await openDB<Schema>(DB_NAME, VERSION, {
      upgrade(d) {
        d.createObjectStore('projects', { keyPath: 'id' });
        d.createObjectStore('samples', { keyPath: 'id' }).createIndex('by-project', 'projectId');
        d.createObjectStore('blobs');
        d.createObjectStore('models', { keyPath: 'id' }).createIndex('by-project', 'projectId');
        d.createObjectStore('modelArtifacts');
        d.createObjectStore('experiments', { keyPath: 'id' }).createIndex('by-project', 'projectId');
        d.createObjectStore('customBackbones', { keyPath: 'id' });
        d.createObjectStore('customBackboneArtifacts');
        d.createObjectStore('settings');
      },
    });
  } catch (e) {
    throw describe(e);
  }

  return {
    kind: 'indexeddb',
    projects: {
      list: () => guard(async () => (await db.getAll('projects')).sort((a, b) => b.updatedAt - a.updatedAt)),
      get: (id) => guard(() => db.get('projects', id)),
      put: (p) => guard(async () => void (await db.put('projects', p))),
      delete: (id) => guard(async () => {
        const tx = db.transaction(['projects', 'samples', 'blobs', 'models', 'modelArtifacts', 'experiments'], 'readwrite');
        for (const key of await tx.objectStore('samples').index('by-project').getAllKeys(id)) {
          await tx.objectStore('samples').delete(key);
          await tx.objectStore('blobs').delete(key);
        }
        for (const key of await tx.objectStore('models').index('by-project').getAllKeys(id)) {
          await tx.objectStore('models').delete(key);
          await tx.objectStore('modelArtifacts').delete(key);
        }
        for (const key of await tx.objectStore('experiments').index('by-project').getAllKeys(id)) await tx.objectStore('experiments').delete(key);
        await tx.objectStore('projects').delete(id);
        await tx.done;
      }),
    },
    samples: {
      list: (pid) => guard(async () => (await db.getAllFromIndex('samples', 'by-project', pid)).sort((a, b) => a.createdAt - b.createdAt)),
      put: (s, b) => guard(async () => {
        const tx = db.transaction(['samples', 'blobs'], 'readwrite');
        await Promise.all([tx.objectStore('samples').put(s), tx.objectStore('blobs').put(b, s.id), tx.done]);
      }),
      getBlob: (id) => guard(async () => (await db.get('blobs', id))?.blob),
      getThumb: (id) => guard(async () => (await db.get('blobs', id))?.thumb),
      delete: (ids) => guard(async () => {
        const tx = db.transaction(['samples', 'blobs'], 'readwrite');
        await Promise.all([...ids.flatMap((id) => [tx.objectStore('samples').delete(id), tx.objectStore('blobs').delete(id)]), tx.done]);
      }),
    },
    models: {
      list: (pid) => guard(async () => {
        const all = pid ? await db.getAllFromIndex('models', 'by-project', pid) : await db.getAll('models');
        return all.sort((a, b) => b.createdAt - a.createdAt);
      }),
      put: (record, artifacts) => guard(async () => {
        const tx = db.transaction(['models', 'modelArtifacts'], 'readwrite');
        await Promise.all([tx.objectStore('models').put(record), tx.objectStore('modelArtifacts').put(artifacts, record.id), tx.done]);
      }),
      get: (id) => guard(async () => {
        const [record, artifacts] = await Promise.all([db.get('models', id), db.get('modelArtifacts', id)]);
        return record && artifacts ? { record, artifacts } : undefined;
      }),
      delete: (id) => guard(async () => {
        const tx = db.transaction(['models', 'modelArtifacts'], 'readwrite');
        await Promise.all([tx.objectStore('models').delete(id), tx.objectStore('modelArtifacts').delete(id), tx.done]);
      }),
    },
customBackbones: {
      list: () => guard(async () => (await db.getAll('customBackbones')).sort((a, b) => b.createdAt - a.createdAt)),
      put: (record, artifacts) => guard(async () => {
        const tx = db.transaction(['customBackbones', 'customBackboneArtifacts'], 'readwrite');
        await Promise.all([tx.objectStore('customBackbones').put(record), tx.objectStore('customBackboneArtifacts').put(artifacts, record.id), tx.done]);
      }),
      get: (id) => guard(async () => {
        const [record, artifacts] = await Promise.all([db.get('customBackbones', id), db.get('customBackboneArtifacts', id)]);
        return record && artifacts ? { record, artifacts } : undefined;
      }),
      delete: (id) => guard(async () => {
        const tx = db.transaction(['customBackbones', 'customBackboneArtifacts'], 'readwrite');
        await Promise.all([tx.objectStore('customBackbones').delete(id), tx.objectStore('customBackboneArtifacts').delete(id), tx.done]);
      }),
    },
    experiments: {
      list: (pid) => guard(async () => (await db.getAllFromIndex('experiments', 'by-project', pid)).sort((a, b) => b.number - a.number)),
      put: (e) => guard(async () => void (await db.put('experiments', e))),
      delete: (id) => guard(async () => void (await db.delete('experiments', id))),
    },
    settings: {
      get: <T>(k: string) => guard(async () => (await db.get('settings', k)) as T | undefined),
      set: (k, v) => guard(async () => void (await db.put('settings', v, k))),
    },
    estimate: async () => {
      if (!navigator.storage?.estimate) return null;
      const e = await navigator.storage.estimate();
      const persisted = navigator.storage.persisted ? await navigator.storage.persisted() : null;
      return { usage: e.usage ?? 0, quota: e.quota ?? 0, persisted };
    },
    requestPersistence: async () => (navigator.storage?.persist ? navigator.storage.persist() : false),
    clearAll: () => guard(async () => {
      const names = ['projects', 'samples', 'blobs', 'models', 'modelArtifacts', 'customBackbones', 'customBackboneArtifacts', 'experiments', 'settings'] as const;
      const tx = db.transaction([...names], 'readwrite');
      await Promise.all([...names.map((n) => tx.objectStore(n).clear()), tx.done]);
    }),
  };
}

/** Opens IndexedDB, or falls back to memory storage and reports why (so the UI can warn the user). */
export async function openStorage(): Promise<{ storage: AppStorage; warning: StorageError | null }> {
  try {
    return { storage: await createIdbStorage(), warning: null };
  } catch (e) {
    const { createMemoryStorage } = await import('./memoryStorage');
    return { storage: createMemoryStorage(), warning: e instanceof StorageError ? e : describe(e) };
  }
}
