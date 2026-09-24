import { create } from 'zustand';
import { analyzeDataset, datasetFingerprint } from '@/ml/data/preflight';
import { buildBundleFiles, readBundle, zipBundle, type NamedBytes } from '@/ml/export/bundle';
import { LIMITS, ModelValidationError, validateLayersArtifacts } from '@/ml/export/validate';
import { hammingHex } from '@/ml/preprocessing/hash';
import { BACKBONES } from '@/ml/models/backboneInfo';
import { openStorage } from '@/storage/idbStorage';
import type { AppStorage } from '@/storage/types';
import type { DatasetClass, ImageSample, IngestOutcome, Project, SampleSource } from '@/types/domain';
import type { BackendPreference, BatchMetrics, CustomBackboneRecord, EpochMetrics, ExperimentRecord, GradientCheckReport, MechanicsTrace, MemoryReport, ModelRecord, RuntimeReport, TrainingConfig } from '@/types/ml';
import type { MlHost, RawSample, TrainPhase, TrainRequest } from '@/types/worker';
import { downloadBytes, fmtBytes, uid } from '@/utils/download';
import { explainError } from '@/utils/errors';
import { startMlHost, type HostInit } from '@/workers/hosts';
import { analyzeImage } from '@/features/dataset/ingest';
import { CLASS_COLORS, DEFAULT_SETTINGS, type AppSettings } from './defaults';

export type PageId = 'dashboard' | 'dataset' | 'training' | 'evaluation' | 'inference' | 'models' | 'settings';

export interface Notice { id: string; kind: 'error' | 'warning' | 'info'; title: string; detail: string; hint?: string }
export interface PerfEntry { at: number; label: string; ms: number }

export interface TrainingState {
  status: 'idle' | 'preparing' | 'training' | 'finalizing' | 'done' | 'error';
  phase: TrainPhase | null;
  phaseDone: number;
  phaseTotal: number;
  phaseMessage: string;
  epochs: EpochMetrics[];
  totalEpochs: number;
  currentEpoch: number;
  batch: BatchMetrics | null;
  mechanics: MechanicsTrace | null;
  warnings: string[];
  startedAt: number | null;
  config: TrainingConfig | null;
  lastExperimentId: string | null;
  backboneProgress: number | null;
}

const IDLE_TRAINING: TrainingState = {
  status: 'idle', phase: null, phaseDone: 0, phaseTotal: 0, phaseMessage: '', epochs: [], totalEpochs: 0, currentEpoch: 0, batch: null,
  mechanics: null, warnings: [], startedAt: null, config: null, lastExperimentId: null, backboneProgress: null,
};

export interface BootDeps {
  storage?: AppStorage;
  startHost?: (pref: BackendPreference) => Promise<HostInit>;
}

interface AppState {
  ready: boolean;
  storage: AppStorage | null;
  host: MlHost | null;
  runtime: RuntimeReport | null;
  fallbackReason: string | null;
  memory: MemoryReport | null;
  page: PageId;
  settings: AppSettings;
  projectPickerOpen: boolean;
  projects: Project[];
  project: Project | null;
  samples: ImageSample[];
  models: ModelRecord[];
  customBackbones: CustomBackboneRecord[];
  experiments: ExperimentRecord[];
  activeModelId: string | null;
  loadedModelId: string | null;
  training: TrainingState;
  ingest: { done: number; total: number } | null;
  notices: Notice[];
  perf: PerfEntry[];
  gradCheck: GradientCheckReport | null;
  storageWarning: string | null;

  boot(deps?: BootDeps): Promise<void>;
  setPage(p: PageId): void;
  closeProjectPicker(): void;
  notify(n: Omit<Notice, 'id'>): void;
  dismissNotice(id: string): void;
  recordPerf(label: string, ms: number): void;
  refreshMemory(): Promise<void>;
  updateSettings(patch: Partial<AppSettings>): Promise<void>;
  restartRuntime(): Promise<void>;

  createProject(name: string): Promise<void>;
  openProject(id: string): Promise<void>;
  deleteProject(id: string): Promise<void>;
  renameProject(name: string): Promise<void>;
  addClass(name?: string): Promise<DatasetClass | null>;
  renameClass(id: string, name: string): Promise<void>;
  deleteClass(id: string): Promise<void>;
  ingestFiles(classId: string, files: File[]): Promise<IngestOutcome[]>;
  ingestBlob(classId: string, blob: Blob, name: string, source: SampleSource, groupId: string, seq: number): Promise<IngestOutcome>;
  deleteSamples(ids: string[]): Promise<void>;

  startTraining(config: TrainingConfig): Promise<void>;
  stopTraining(): void;
  runGradientCheck(): Promise<void>;
  loadModel(id: string): Promise<boolean>;
  deleteModel(id: string): Promise<void>;
  exportModel(id: string, kind: 'zip' | 'files'): Promise<void>;
  importModel(files: File[]): Promise<void>;
  importCustomBackbone(files: File[], name?: string): Promise<void>;
  deleteCustomBackbone(id: string): Promise<void>;
  deleteExperiment(id: string): Promise<void>;
  clearAllData(): Promise<void>;
}

let unsubscribers: (() => void)[] = [];

export const useApp = create<AppState>((set, get) => {
  const notifyErr = (e: unknown, ctx?: string) => {
    const x = explainError(e, ctx);
    get().notify({ kind: 'error', title: x.title, detail: x.detail, hint: x.hint });
  };
  const saveProject = async (p: Project) => {
    const next = { ...p, updatedAt: Date.now() };
    await get().storage?.projects.put(next);
    set({ project: next, projects: get().projects.map((x) => (x.id === next.id ? next : x)) });
  };

  const bindHost = (host: MlHost) => {
    unsubscribers.forEach((u) => u());
    const patch = (fn: (t: TrainingState) => Partial<TrainingState>) => set((s) => ({ training: { ...s.training, ...fn(s.training) } }));
    unsubscribers = [
      host.on('train:phase', (p) => patch(() => ({ phase: p.phase, phaseDone: p.done ?? 0, phaseTotal: p.total ?? 0, phaseMessage: p.message ?? '', status: p.phase === 'training' ? 'training' : p.phase === 'finalizing' || p.phase === 'evaluating' ? 'finalizing' : 'preparing' }))),
      host.on('train:epochStart', (e) => patch(() => ({ currentEpoch: e.epoch, totalEpochs: e.epochs }))),
      host.on('train:batch', (b) => patch(() => ({ batch: b }))),
      host.on('train:epoch', (m) => patch((t) => ({ epochs: [...t.epochs, m] }))),
      host.on('train:mechanics', (m) => patch(() => ({ mechanics: m }))),
      host.on('backbone:progress', (p) => patch(() => ({ backboneProgress: p.fraction >= 1 ? null : p.fraction }))),
    ];
  };

  const applyHost = (h: HostInit) => {
    bindHost(h.host);
    set({ host: h.host, runtime: h.report, fallbackReason: h.fallbackReason, loadedModelId: null });
    if (h.fallbackReason) get().notify({ kind: 'warning', title: 'Running the ML runtime on the main thread', detail: h.fallbackReason, hint: 'Training still works, but the page may stutter while it runs.' });
  };

  const loadProjectData = async (project: Project) => {
    const st = get().storage!;
    const [samples, models, experiments, customBackbones] = await Promise.all([st.samples.list(project.id), st.models.list(), st.experiments.list(project.id), st.customBackbones.list()]);
    set({ project, samples, models, experiments, customBackbones, activeModelId: models[0]?.id ?? null });
  };

  return {
    ready: false, storage: null, host: null, runtime: null, fallbackReason: null, memory: null, page: 'dashboard', settings: DEFAULT_SETTINGS,
    projectPickerOpen: false,
    projects: [], project: null, samples: [], models: [], customBackbones: [], experiments: [], activeModelId: null, loadedModelId: null,
    training: IDLE_TRAINING, ingest: null, notices: [], perf: [], gradCheck: null, storageWarning: null,

    async boot(deps) {
      let storage = deps?.storage ?? null;
      if (!storage) {
        const opened = await openStorage();
        storage = opened.storage;
        if (opened.warning) set({ storageWarning: `${opened.warning.message} ${opened.warning.hint}` });
      }
      const saved = (await storage.settings.get<AppSettings>('app-settings')) ?? DEFAULT_SETTINGS;
      const settings = { ...DEFAULT_SETTINGS, ...saved };
      set({ storage, settings });
      let projects = await storage.projects.list();
      if (projects.length === 0) {
        const p: Project = { id: uid(), name: 'My first project', createdAt: Date.now(), updatedAt: Date.now(), classes: [] };
        p.classes = [1, 2].map((n, i) => ({ id: uid(), name: `Class ${n}`, color: CLASS_COLORS[i] as string, createdAt: Date.now() }));
        await storage.projects.put(p);
        projects = [p];
      }
      set({ projects });
      await loadProjectData(projects[0] as Project);
      set({ ready: true, projectPickerOpen: true });
      try {
        applyHost(await (deps?.startHost ?? startMlHost)(settings.backend));
        get().recordPerf('Runtime init', 0);
      } catch (e) {
        notifyErr(e, 'ML runtime could not start');
      }
    },

    setPage: (page) => set({ page }),
    closeProjectPicker: () => set({ projectPickerOpen: false }),
    notify: (n) => set((s) => ({ notices: [...s.notices.slice(-3), { ...n, id: uid() }] })),
    dismissNotice: (id) => set((s) => ({ notices: s.notices.filter((n) => n.id !== id) })),
    recordPerf: (label, ms) => set((s) => ({ perf: [{ at: Date.now(), label, ms }, ...s.perf].slice(0, 60) })),
    async refreshMemory() {
      const h = get().host;
      if (h) set({ memory: await h.memory().catch(() => null) });
    },

    async updateSettings(patch) {
      const settings = { ...get().settings, ...patch };
      set({ settings });
      await get().storage?.settings.set('app-settings', settings);
      if (patch.backend && patch.backend !== get().runtime?.activeBackend) await get().restartRuntime();
    },

    async restartRuntime() {
      if (['preparing', 'training', 'finalizing'].includes(get().training.status)) {
        get().notify({ kind: 'warning', title: 'Training in progress', detail: 'The runtime cannot be restarted while training.', hint: 'Stop training first.' });
        return;
      }
      get().host?.terminate();
      set({ host: null, runtime: null, loadedModelId: null });
      try {
        applyHost(await startMlHost(get().settings.backend));
      } catch (e) {
        notifyErr(e, 'ML runtime could not start');
      }
    },

    async createProject(name) {
      const p: Project = { id: uid(), name: name.trim() || 'Untitled project', createdAt: Date.now(), updatedAt: Date.now(), classes: [] };
      p.classes = [1, 2].map((n, i) => ({ id: uid(), name: `Class ${n}`, color: CLASS_COLORS[i] as string, createdAt: Date.now() }));
      await get().storage!.projects.put(p);
      set({ projects: [p, ...get().projects] });
      await loadProjectData(p);
    },
    async openProject(id) {
      const p = get().projects.find((x) => x.id === id);
      if (p) await loadProjectData(p);
    },
    async deleteProject(id) {
      await get().storage!.projects.delete(id);
      const projects = get().projects.filter((p) => p.id !== id);
      if (projects.length === 0) {
        set({ projects: [] });
        await get().createProject('My first project');
      } else {
        set({ projects });
        if (get().project?.id === id) await loadProjectData(projects[0] as Project);
      }
    },
    async renameProject(name) {
      const p = get().project;
      if (p) await saveProject({ ...p, name: name.trim() || p.name });
    },

    async addClass(name) {
      const p = get().project;
      if (!p) return null;
      const c: DatasetClass = { id: uid(), name: name?.trim() || `Class ${p.classes.length + 1}`, color: CLASS_COLORS[p.classes.length % CLASS_COLORS.length] as string, createdAt: Date.now() };
      await saveProject({ ...p, classes: [...p.classes, c] });
      return c;
    },
    async renameClass(id, name) {
      const p = get().project;
      if (p) await saveProject({ ...p, classes: p.classes.map((c) => (c.id === id ? { ...c, name: name.trim() || c.name } : c)) });
    },
    async deleteClass(id) {
      const p = get().project;
      if (!p) return;
      await get().deleteSamples(get().samples.filter((s) => s.classId === id).map((s) => s.id));
      await saveProject({ ...p, classes: p.classes.filter((c) => c.id !== id) });
    },

    async ingestBlob(classId, blob, name, source, groupId, seq) {
      const { project, storage } = get();
      if (!project || !storage) return { name, accepted: false, reason: 'corrupted', detail: 'No project is open.', warnings: [] };
      const a = await analyzeImage(blob);
      if (!a.ok) return { name, accepted: false, reason: a.reason, detail: a.detail, warnings: [] };
      const existing = get().samples;
      const same = existing.find((s) => s.sha256 === a.sha256);
      if (same) {
        const otherClass = same.classId !== classId;
        return {
          name, accepted: false, warnings: [], reason: otherClass ? 'label-conflict' : 'duplicate',
          detail: otherClass
            ? `Identical to an image already in "${project.classes.find((c) => c.id === same.classId)?.name ?? 'another class'}". The same image cannot have two labels.`
            : 'This exact image is already in this class.',
        };
      }
      if (source === 'upload' && existing.some((s) => hammingHex(s.phash, a.phash) <= 4)) a.quality.warnings.push('near-duplicate');
      const sample: ImageSample = {
        id: uid(), projectId: project.id, classId, source, name, width: a.width, height: a.height, mimeType: a.mimeType, size: a.size, createdAt: Date.now() + seq,
        groupId, seq, sha256: a.sha256, phash: a.phash, quality: a.quality,
      };
      try {
        await storage.samples.put(sample, { blob, thumb: a.thumb });
      } catch (e) {
        notifyErr(e);
        return { name, accepted: false, reason: 'corrupted', detail: 'Could not be saved to browser storage.', warnings: [] };
      }
      set((s) => ({ samples: [...s.samples, sample] }));
      return { name, accepted: true, warnings: a.quality.warnings, sampleId: sample.id };
    },

    async ingestFiles(classId, files) {
      const outcomes: IngestOutcome[] = [];
      set({ ingest: { done: 0, total: files.length } });
      const t0 = performance.now();
      for (let i = 0; i < files.length; i++) {
        const f = files[i] as File;
        outcomes.push(await get().ingestBlob(classId, f, f.name, 'upload', uid(), 0));
        set({ ingest: { done: i + 1, total: files.length } });
      }
      set({ ingest: null });
      get().recordPerf(`Dataset import (${files.length} files)`, performance.now() - t0);
      return outcomes;
    },

    async deleteSamples(ids) {
      if (!ids.length) return;
      await get().storage!.samples.delete(ids);
      const gone = new Set(ids);
      set((s) => ({ samples: s.samples.filter((x) => !gone.has(x.id)) }));
    },

    async startTraining(config) {
      const { host, project, samples, storage } = get();
      if (!host || !project || !storage) return notifyErr(new Error('The ML runtime is not running.'), 'Cannot train');
      const analysis = analyzeDataset(project.classes, samples, config);
      if (analysis.blocking) return get().notify({ kind: 'error', title: 'Dataset is not ready', detail: analysis.issues.filter((i) => i.severity === 'error').map((i) => i.message).join(' '), hint: 'Add more samples on the Dataset page.' });

      set({ training: { ...IDLE_TRAINING, status: 'preparing', startedAt: Date.now(), config, totalEpochs: config.epochs, phase: 'decoding' }, loadedModelId: null });
      try {
        const used = project.classes.filter((c) => samples.some((s) => s.classId === c.id));
        const usedIds = new Set(used.map((c) => c.id));
        const raw: RawSample[] = [];
        for (const s of samples) {
          if (!usedIds.has(s.classId)) continue;
          const blob = await storage.samples.getBlob(s.id);
          if (blob) raw.push({ id: s.id, classId: s.classId, groupId: s.groupId, seq: s.seq, blob });
        }
        let customBackbone: TrainRequest['customBackbone'] = null;
        if (config.mode === 'transfer' && config.customBackboneId) {
          const stored = await storage.customBackbones.get(config.customBackboneId);
          if (!stored) throw new Error(`The selected custom backbone is no longer available (it may have been deleted). Pick another backbone.`);
          customBackbone = { id: stored.record.id, name: stored.record.name, artifacts: stored.artifacts };
        }
        const res = await host.train({ classes: used.map((c) => ({ id: c.id, name: c.name })), samples: raw, config, customBackbone });

        const number = (get().experiments[0]?.number ?? 0) + 1;
        const backboneLabel = config.customBackboneId ? (get().customBackbones.find((b) => b.id === config.customBackboneId)?.name ?? 'custom backbone') : BACKBONES[config.backboneId].label;
        const arch = config.mode === 'transfer' ? `${backboneLabel} + Dense(${config.hiddenUnits})` : `CNN ${config.scratchFilters.join('-')} @${config.scratchInputSize}px`;
        const modelId = uid();
        const experiment: ExperimentRecord = {
          id: uid(), number, projectId: project.id, createdAt: Date.now(), datasetHash: datasetFingerprint(samples.filter((s) => usedIds.has(s.classId))),
          sampleCount: raw.length, classNames: used.map((c) => c.name), modelType: res.metadata.modelType, architecture: arch, config, backend: res.backend,
          epochsRun: res.history.length, history: res.history, finalMetrics: res.metadata.finalMetrics, durationMs: res.durationMs, paramCount: res.paramCount,
          modelBytes: res.modelBytes, timings: res.timings, modelId, trainIds: res.trainIds, valIds: res.valIds, evaluation: res.evaluation,
          stoppedEarly: res.stoppedEarly, aborted: res.aborted,
        };
        const record: ModelRecord = { id: modelId, projectId: project.id, name: `${arch} · #${String(number).padStart(3, '0')}`, createdAt: Date.now(), metadata: res.metadata, experimentId: experiment.id, sizeBytes: res.modelBytes, imported: false };
        await storage.models.put(record, res.artifacts);
        await storage.experiments.put(experiment);
        set((s) => ({
          experiments: [experiment, ...s.experiments], models: [record, ...s.models], activeModelId: modelId, loadedModelId: modelId,
          training: { ...s.training, status: 'done', warnings: res.warnings, lastExperimentId: experiment.id, phase: null },
        }));
        const t = res.timings;
        (Object.entries({ 'Dataset decode + preprocess': t.decodeMs, 'Backbone load': t.backboneLoadMs, 'Feature extraction': t.featureExtractionMs, 'Training': t.trainMs, 'Evaluation': t.evaluateMs, 'Total run': t.totalMs }) as [string, number | undefined][])
          .forEach(([k, v]) => v !== undefined && v > 0 && get().recordPerf(k, v));
        void get().refreshMemory();
      } catch (e) {
        const x = explainError(e, 'Training failed');
        set((s) => ({ training: { ...s.training, status: 'error', phase: null } }));
        get().notify({ kind: 'error', title: x.title, detail: x.detail, hint: x.hint });
      }
    },

    stopTraining: () => get().host?.abortTraining(),

    async runGradientCheck() {
      try {
        const t0 = performance.now();
        set({ gradCheck: await get().host!.gradientCheck() });
        get().recordPerf('Gradient check', performance.now() - t0);
      } catch (e) {
        notifyErr(e, 'Gradient check failed');
      }
    },

    async loadModel(id) {
      const { host, storage } = get();
      if (!host || !storage) return false;
      if (get().loadedModelId === id) return true;
      try {
        const stored = await storage.models.get(id);
        if (!stored) throw new Error('The model is missing from browser storage.');
        let customBackbone: TrainRequest['customBackbone'] = null;
        if (stored.record.metadata.customBackbone) {
          const cb = await storage.customBackbones.get(stored.record.metadata.customBackbone.id);
          if (!cb) throw new Error(`This model needs its custom backbone "${stored.record.metadata.customBackbone.name}", which is no longer in browser storage.`);
          customBackbone = { id: cb.record.id, name: cb.record.name, artifacts: cb.artifacts };
        }
        const r = await host.loadModel(stored.artifacts, stored.record.metadata, customBackbone);
        get().recordPerf(`Model load (${stored.record.name})`, r.loadMs);
        set({ loadedModelId: id, activeModelId: id });
        return true;
      } catch (e) {
        notifyErr(e, 'Could not load the model');
        return false;
      }
    },

    async deleteModel(id) {
      await get().storage!.models.delete(id);
      if (get().loadedModelId === id) {
        await get().host?.unloadModel();
        set({ loadedModelId: null });
      }
      set((s) => ({ models: s.models.filter((m) => m.id !== id), activeModelId: s.activeModelId === id ? null : s.activeModelId }));
    },

    async exportModel(id, kind) {
      try {
        const stored = await get().storage!.models.get(id);
        if (!stored) throw new Error('The model is missing from browser storage.');
        const files = buildBundleFiles(stored.artifacts, stored.record.metadata);
        const base = stored.record.name.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
        if (kind === 'zip') downloadBytes(`${base}.zip`, zipBundle(files), 'application/zip');
        else {
          downloadBytes('model.json', files['model.json'], 'application/json');
          downloadBytes('weights.bin', files['weights.bin']);
          downloadBytes('metadata.json', files['metadata.json'], 'application/json');
        }
      } catch (e) {
        notifyErr(e, 'Export failed');
      }
    },

    async importModel(files) {
      try {
        const inputs: NamedBytes[] = await Promise.all(files.map(async (f) => ({ name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) })));
        const { artifacts, metadata } = readBundle(inputs);
        const record: ModelRecord = {
          id: uid(), projectId: null, name: `Imported · ${metadata.modelType === 'scratch-cnn' ? 'CNN' : 'MobileNetV2'} · ${metadata.classes.length} classes`,
          createdAt: Date.now(), metadata, experimentId: null, sizeBytes: artifacts.weightData.byteLength, imported: true,
        };
        await get().storage!.models.put(record, artifacts);
        set((s) => ({ models: [record, ...s.models], activeModelId: record.id }));
        get().notify({ kind: 'info', title: 'Model imported', detail: `${metadata.classes.map((c) => c.name).join(', ')} — ${metadata.preprocessing.inputSize}px input, ${metadata.modelType}.`, hint: 'Preprocessing was restored from the model\'s own metadata.' });
      } catch (e) {
        notifyErr(e, 'Model import failed');
      }
    },

    async importCustomBackbone(files, name) {
      try {
        const inputs = await Promise.all(files.map(async (f) => ({ name: f.name, bytes: new Uint8Array(await f.arrayBuffer()) })));
        const modelJsonFile = inputs.find((i) => i.name.toLowerCase().endsWith('.json') && !i.name.toLowerCase().includes('metadata'));
        const weightsFile = inputs.find((i) => i.name.toLowerCase().endsWith('.bin')) ?? inputs.find((i) => !i.name.toLowerCase().endsWith('.json'));
        if (!modelJsonFile) throw new ModelValidationError('No model.json file was selected.', 'Select a TF.js layers-format model.json together with its weights.bin.');
        if (!weightsFile) throw new ModelValidationError('No weights file was selected.', 'Select model.json together with its weights.bin.');
        if (modelJsonFile.bytes.length > LIMITS.modelJsonBytes) throw new ModelValidationError('model.json exceeds the size limit.', 'Import a smaller model.');
        let modelJson: { modelTopology?: unknown; weightsManifest?: { weights?: unknown[] }[] };
        try {
          modelJson = JSON.parse(new TextDecoder().decode(modelJsonFile.bytes));
        } catch {
          throw new ModelValidationError('model.json could not be parsed.', 'The file is corrupted or not JSON.');
        }
        if (!modelJson.modelTopology || !Array.isArray(modelJson.weightsManifest)) throw new ModelValidationError('model.json is not a recognisable TF.js layers-model.', 'A GraphModel (frozen SavedModel/TFHub format) cannot be used here — only Sequential/functional Keras layers-models.');
        const weightSpecs = modelJson.weightsManifest.flatMap((g) => g.weights ?? []);
        const artifacts = { modelTopology: modelJson.modelTopology, weightSpecs, weightData: weightsFile.bytes.buffer.slice(weightsFile.bytes.byteOffset, weightsFile.bytes.byteOffset + weightsFile.bytes.byteLength) as ArrayBuffer };
        validateLayersArtifacts(artifacts); // layer allow-list + weight byte-count check on this untrusted file
        const { loadCustomBackbone } = await import('@/ml/models/backbone');
        const probe = await loadCustomBackbone(uid(), artifacts); // also verifies it actually loads and has a usable output shape
        const record: CustomBackboneRecord = { id: uid(), name: name?.trim() || modelJsonFile.name.replace(/\.json$/i, ''), createdAt: Date.now(), featureDim: probe.featureDim, inputSize: probe.spec.inputSize, sizeBytes: artifacts.weightData.byteLength };
        probe.dispose();
        await get().storage!.customBackbones.put(record, artifacts);
        set((s) => ({ customBackbones: [record, ...s.customBackbones] }));
        get().notify({ kind: 'info', title: 'Custom backbone imported', detail: `"${record.name}" — ${record.featureDim}-d features, ${fmtBytes(record.sizeBytes)}. Available under Training → Transfer learning → Backbone.` });
      } catch (e) {
        notifyErr(e, 'Custom backbone import failed');
      }
    },

    async deleteCustomBackbone(id) {
      await get().storage!.customBackbones.delete(id);
      set((s) => ({ customBackbones: s.customBackbones.filter((b) => b.id !== id) }));
    },

    async deleteExperiment(id) {
      await get().storage!.experiments.delete(id);
      set((s) => ({ experiments: s.experiments.filter((e) => e.id !== id) }));
    },

    async clearAllData() {
      get().host?.abortTraining();
      await get().host?.unloadModel().catch(() => undefined);
      await get().storage!.clearAll();
      set({ projects: [], project: null, samples: [], models: [], customBackbones: [], experiments: [], activeModelId: null, loadedModelId: null, training: IDLE_TRAINING });
      await get().createProject('My first project');
    },
  };
});
