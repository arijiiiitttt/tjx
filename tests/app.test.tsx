// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

vi.mock('@/features/dataset/ingest', () => ({
  // Deterministic stand-in for canvas-based analysis (jsdom has no createImageBitmap): the blob's text drives the outcome.
  analyzeImage: async (blob: Blob) => {
    const text = await blob.text();
    if (text === 'corrupt') return { ok: false, reason: 'corrupted', detail: 'The file could not be decoded as an image.' };
    if (text === 'blank') return { ok: false, reason: 'blank', detail: 'The image has almost no variation.' };
    const [sha, phash = '0000000000000000'] = text.split('|');
    return { ok: true, width: 640, height: 480, mimeType: 'image/jpeg', size: blob.size, sha256: sha, phash, quality: { meanLuma: 100, stdLuma: 40, sharpness: 100, warnings: [] }, thumb: new Blob(['t']) };
  },
  MAX_FILE_BYTES: 1, THUMB_SIZE: 128,
}));

import { App } from '@/app/App';
import { useApp } from '@/app/store';
import { createMemoryStorage } from '@/storage/memoryStorage';
import type { AppStorage } from '@/storage/types';
import type { ImageSample } from '@/types/domain';
import type { MlEvents, MlHost, TrainRequest, TrainResponse } from '@/types/worker';
import type { RuntimeReport } from '@/types/ml';

beforeAll(() => {
  if (!globalThis.crypto?.randomUUID) Object.defineProperty(globalThis, 'crypto', { value: nodeCrypto(), configurable: true });
  URL.createObjectURL = () => 'blob:mock';
  URL.revokeObjectURL = () => undefined;
});
function nodeCrypto() {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require('node:crypto').webcrypto;
}

const report: RuntimeReport = { activeBackend: 'webgl', attempts: [{ backend: 'webgpu', ok: false, error: 'navigator.gpu is not available' }, { backend: 'webgl', ok: true, probeMs: 12 }], affineAugmentation: true, affineBatch: true, host: 'worker', gpu: null };

function fakeHost(over: Partial<MlHost> = {}) {
  const listeners = new Map<string, Set<(p: never) => void>>();
  const emit = <K extends keyof MlEvents>(n: K, p: MlEvents[K]) => listeners.get(n)?.forEach((cb) => cb(p as never));
  const trainCalls: TrainRequest[] = [];
  const host: MlHost = {
    kind: 'worker',
    init: async () => report,
    memory: async () => ({ numTensors: 12, numBytes: 4096, numBytesInGPU: null, unreliable: false }),
    train: async (req) => {
      trainCalls.push(req);
      emit('train:phase', { phase: 'training' });
      for (let e = 1; e <= 3; e++) {
        emit('train:epochStart', { epoch: e, epochs: 3 });
        emit('train:epoch', { epoch: e, loss: 1 / e, accuracy: 0.5 + e * 0.1, valLoss: 1.2 / e, valAccuracy: 0.5 + e * 0.1, epochMs: 40, samplesPerSec: 900 });
      }
      const ids = req.samples.map((s) => s.id);
      const res: TrainResponse = {
        metadata: {
          schemaVersion: 1, modelType: 'scratch-cnn', classes: req.classes, inputShape: [32, 32, 3], modelInputShape: [32, 32, 3],
          preprocessing: { version: 1, inputSize: 32, resize: 'center-crop', colorSpace: 'rgb', normalization: { min: 0, max: 1 } }, backbone: null, customBackbone: null, outputActivation: 'softmax',
          trainingConfig: req.config, finalMetrics: { loss: 0.3, accuracy: 0.8, valLoss: 0.4, valAccuracy: 0.8, macroF1: 0.79 }, createdAt: new Date().toISOString(), appVersion: '0.1.0',
        },
        artifacts: { modelTopology: {}, weightSpecs: [], weightData: new ArrayBuffer(16) },
        evaluation: { sampleCount: 4, accuracy: 0.75, macroF1: 0.7, weightedF1: 0.7, perClass: req.classes.map((c) => ({ classId: c.id, className: c.name, precision: 0.7, recall: 0.7, f1: 0.7, support: 2 })), confusion: [[2, 0], [1, 1]], classIds: req.classes.map((c) => c.id), classNames: req.classes.map((c) => c.name), misclassified: [] },
        history: [], trainIds: ids.slice(0, 8), valIds: ids.slice(8), purgedIds: [], failedSampleIds: [], durationMs: 1234, paramCount: 4321, modelBytes: 16, timings: { trainMs: 1000, totalMs: 1234 }, backend: 'webgl', stoppedEarly: false, aborted: false, warnings: [],
      };
      return res;
    },
    loadModel: async () => ({ loadMs: 5 }), unloadModel: async () => undefined,
    predictBitmap: async () => ({ predictions: [], latencyMs: 1, preprocessMs: 1 }),
    evaluate: async () => { throw new Error('unused'); }, gradientCheck: async () => { throw new Error('unused'); }, prefetchBackbone: async () => ({ cached: true, ms: 0 }),
    abortTraining: () => undefined, terminate: () => undefined,
    on: (name, cb) => { const set = listeners.get(name) ?? new Set(); set.add(cb as never); listeners.set(name, set); return () => set.delete(cb as never); },
    ...over,
  };
  return { host, trainCalls };
}

async function seed(storage: AppStorage, perClass: [number, number]) {
  const project = { id: 'p1', name: 'Test project', createdAt: 1, updatedAt: 1, classes: [{ id: 'c1', name: 'Cat', color: '#fff', createdAt: 1 }, { id: 'c2', name: 'Dog', color: '#000', createdAt: 1 }] };
  await storage.projects.put(project);
  let t = 0;
  for (const [ci, n] of perClass.entries()) {
    for (let i = 0; i < n; i++) {
      const s: ImageSample = {
        id: `s${ci}-${i}`, projectId: 'p1', classId: `c${ci + 1}`, source: 'upload', name: `img-${ci}-${i}.jpg`, width: 640, height: 480, mimeType: 'image/jpeg', size: 1000, createdAt: ++t, groupId: `g${ci}-${i}`, seq: 0,
        sha256: `sha-${ci}-${i}`, phash: '0', quality: { meanLuma: 100, stdLuma: 40, sharpness: 100, warnings: [] },
      };
      await storage.samples.put(s, { blob: new Blob([`b${ci}${i}`]), thumb: new Blob(['t']) });
    }
  }
}

async function mount(perClass: [number, number], hostOver: Partial<MlHost> = {}) {
  const storage = createMemoryStorage();
  await seed(storage, perClass);
  const { host, trainCalls } = fakeHost(hostOver);
  useApp.setState(useApp.getInitialState(), true); // the store is module-level state: reset between tests
  const deps = { storage, startHost: async () => ({ host, report, fallbackReason: null }) };
  render(<App deps={deps} />);
  await screen.findByRole('heading', { name: 'Test project' });
  await dismissProjectPicker();
  return { storage, host, trainCalls };
}
const go = async (name: string) => {
  if (name === 'Dashboard' || name === 'Settings') fireEvent.click(screen.getByTitle(name));
  else fireEvent.click(within(screen.getByRole('list', { name: 'Guided process' })).getByRole('button', { name }));
};
/** The step-picker dialog blocks everything on first load; dismiss it by continuing with the seeded project. */
const dismissProjectPicker = async () => {
  fireEvent.click(await screen.findByRole('button', { name: /Continue with/ }));
  await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
};

afterEach(cleanup);

describe('application shell', () => {
  it('boots, shows the privacy statement and live runtime info, and every page renders', async () => {
    await mount([12, 12]);
    expect(screen.getByText(/Your images stay on your device/)).toBeTruthy();
    await waitFor(() => expect(screen.getAllByText('webgl').length).toBeGreaterThan(0));
    await go('Dataset');
    await screen.findByRole('heading', { level: 1, name: 'Dataset' });
    await go('Training');
    await screen.findByRole('heading', { level: 1, name: 'Training' });
    const train = () => screen.getByRole('button', { name: /Train model/ }) as HTMLButtonElement;
    await screen.findByRole('button', { name: /Train model/ });
    fireEvent.click(screen.getByLabelText(/I understand these warnings/)); // small-class-size warning, since 12 < the 30 recommended
    await waitFor(() => expect(train().disabled).toBe(false));
    fireEvent.click(train());
    await screen.findByText('Experiment #001'); // completing a run unlocks Evaluation, Inference and Models
    for (const [nav, heading] of [['Evaluation', 'Evaluation'], ['Inference', 'Inference'], ['Models', 'Models & experiments'], ['Settings', 'Settings'], ['Dashboard', 'Test project']] as const) {
      await go(nav);
      await screen.findByRole('heading', { level: 1, name: heading });
    }
  });

  it('dataset page reports class imbalance and blocks training for tiny classes', async () => {
    await mount([40, 12]);
    await go('Dataset');
    await screen.findByText('\u26a0 Class imbalance detected');
    expect(screen.getAllByText(/Cat: 40/).length).toBeGreaterThan(0);
    cleanup();
    await mount([12, 3]);
    await go('Dataset');
    await screen.findByText(/Not trainable yet/);
    // the dataset isn't trainable yet, so the Training tab and the "Continue" button stay locked
    expect((within(screen.getByRole('list', { name: 'Guided process' })).getByRole('button', { name: 'Training' }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByRole('button', { name: /Continue to Training/ }) as HTMLButtonElement).disabled).toBe(true);
  });
});

describe('guided-process gating', () => {
  it('lets a finished step be revisited, but keeps later steps locked until their prerequisites are done', async () => {
    await mount([12, 3]); // "Dog" has too few samples to train
    const trainingTab = () => within(screen.getByRole('list', { name: 'Guided process' })).getByRole('button', { name: 'Training' }) as HTMLButtonElement;
    expect(trainingTab().disabled).toBe(true);
    await act(async () => {
      await useApp.getState().ingestFiles(useApp.getState().project!.classes[1]!.id, [new File(['x|0000000000000000'], 'x.jpg')]);
      await useApp.getState().ingestFiles(useApp.getState().project!.classes[1]!.id, [new File(['y|0000000000000001'], 'y.jpg')]);
    });
    await waitFor(() => expect(trainingTab().disabled).toBe(false)); // Dog now has 5 samples: dataset step is done
    await go('Training');
    await screen.findByRole('heading', { level: 1, name: 'Training' });
    expect(within(screen.getByRole('list', { name: 'Guided process' })).getByRole('button', { name: 'Evaluation' }).hasAttribute('disabled')).toBe(true); // no model trained yet
    await go('Dataset'); // going back is always allowed
    await screen.findByRole('heading', { level: 1, name: 'Dataset' });
  });
});

describe('training flow through the real store', () => {
  it('requires acknowledging warnings, sends the right request, records an experiment and a model', async () => {
    const { trainCalls, storage } = await mount([40, 12]);
    await go('Training');
    await screen.findByRole('button', { name: /Train model/ });
    const train = () => screen.getByRole('button', { name: /Train model/ }) as HTMLButtonElement;
    await waitFor(() => expect(train().disabled).toBe(true)); // imbalance warning must be acknowledged
    fireEvent.click(screen.getByLabelText(/I understand these warnings/));
    await waitFor(() => expect(train().disabled).toBe(false));
    fireEvent.click(train());

    await screen.findByText('Experiment #001');
    expect(trainCalls).toHaveLength(1);
    const req = trainCalls[0]!;
    expect(req.samples).toHaveLength(52);
    expect(req.samples.every((s) => s.blob instanceof Blob)).toBe(true);
    expect(req.classes.map((c) => c.name)).toEqual(['Cat', 'Dog']);
    expect(req.config.mode).toBe('transfer');
    expect(req.config.classWeighting).toBe('balanced');

    const state = useApp.getState();
    expect(state.experiments).toHaveLength(1);
    expect(state.experiments[0]!.datasetHash).toMatch(/^[0-9a-f]{8}$/);
    expect(state.loadedModelId).toBe(state.models[0]!.id);
    expect((await storage.models.list()).length).toBe(1);
    expect((await storage.experiments.list('p1')).length).toBe(1);

    await go('Models');
    await screen.findByText(/#001/);
    expect(screen.getAllByText(/Test project|Cat · Dog/).length).toBeGreaterThan(0);
  });

  it('shows an actionable error when the GPU runs out of memory', async () => {
    await mount([40, 40], { train: async () => { throw new Error('Failed to allocate texture: out of memory'); } });
    await go('Training');
    fireEvent.click(await screen.findByRole('button', { name: /Train model/ }));
    await screen.findByText('The GPU ran out of memory');
    expect(screen.getByText(/Lower the batch size/)).toBeTruthy();
    expect(useApp.getState().training.status).toBe('error');
  });
});

describe('dataset ingestion rules', () => {
  it('rejects corrupted/blank images, exact duplicates and cross-class label conflicts; warns on near-duplicates', async () => {
    await mount([0, 0]);
    const st = () => useApp.getState();
    const [cat, dog] = st().project!.classes;
    const ingest = (classId: string, text: string) => st().ingestBlob(classId, new Blob([text]), `${text}.jpg`, 'upload', `g-${text}`, 0);
    expect((await ingest(cat!.id, 'corrupt')).reason).toBe('corrupted');
    expect((await ingest(cat!.id, 'blank')).reason).toBe('blank');
    expect((await ingest(cat!.id, 'aaa|0000000000000000')).accepted).toBe(true);
    expect((await ingest(cat!.id, 'aaa|0000000000000000')).reason).toBe('duplicate');
    const conflict = await ingest(dog!.id, 'aaa|0000000000000000');
    expect(conflict.reason).toBe('label-conflict');
    expect(conflict.detail).toContain('Cat');
    const near = await ingest(dog!.id, 'bbb|0000000000000001'); // hamming distance 1 from an existing image
    expect(near.accepted).toBe(true);
    expect(near.warnings).toContain('near-duplicate');
    expect(st().samples).toHaveLength(2);
    // persisted, with the original bytes untouched
    const stored = await st().storage!.samples.getBlob(near.sampleId!);
    expect(await stored!.text()).toBe('bbb|0000000000000001');
  });

  it('deleting a class removes its samples', async () => {
    await mount([6, 6]);
    await act(async () => { await useApp.getState().deleteClass('c1'); });
    expect(useApp.getState().samples.every((s) => s.classId === 'c2')).toBe(true);
    expect(useApp.getState().project!.classes).toHaveLength(1);
  });
});
