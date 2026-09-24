import * as tf from '@tensorflow/tfjs';
import type { BackboneId, BackendPreference, EvaluationReport, GradientCheckReport, MemoryReport, ModelArtifactsData, ModelMetadata, PerfTimings, PredictionResult, PreprocessingSpec, RuntimeReport } from '@/types/ml';
import type { CustomBackboneInput, MlApi, MlEvents, RawSample, TrainRequest, TrainResponse } from '@/types/worker';
import { decodeSamples } from './data/prepare';
import { splitDataset } from './data/split';
import { Predictor } from './inference/predictor';
import { buildEvaluationReport } from './metrics/classification';
import { BACKBONES, backboneRef, loadBackbone, loadCustomBackbone, type FeatureExtractor } from './models/backbone';
import { buildCnn } from './models/cnn';
import { buildHeadModel, type ClassifierModel } from './models/head';
import { scratchSpec } from './preprocessing/spec';
import { bitmapToPixels } from './preprocessing/decode';
import { initBackend, memoryReport } from './runtime/backend';
import { evaluateSplit, trainClassifier, type AbortToken } from './training/engine';
import { gradientCheck } from './training/gradcheck';
import { buildFeatureData, buildImageData } from './training/pipeline';
import { validateArtifacts } from './export/validate';
import { APP_VERSION } from './version';

const now = () => performance.now();

type Emit = <K extends keyof MlEvents>(name: K, payload: MlEvents[K]) => void;

function concatBuffers(w: ArrayBuffer | ArrayBuffer[]): ArrayBuffer {
  if (!Array.isArray(w)) return w;
  const total = w.reduce((a, b) => a + b.byteLength, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of w) {
    out.set(new Uint8Array(b), off);
    off += b.byteLength;
  }
  return out.buffer;
}

export async function extractArtifacts(model: tf.LayersModel): Promise<ModelArtifactsData> {
  let captured: tf.io.ModelArtifacts | undefined;
  await model.save(
    tf.io.withSaveHandler(async (a) => {
      captured = a;
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: 'JSON' as const } };
    }),
  );
  if (!captured?.weightData || !captured.weightSpecs) throw new Error('Model serialisation produced no weights.');
  return { modelTopology: captured.modelTopology, weightSpecs: captured.weightSpecs, weightData: concatBuffers(captured.weightData) };
}

/**
 * The ML runtime. It is environment-agnostic: it runs inside a Web Worker in production and on the
 * main thread as a fallback / in tests. It never touches the DOM or React.
 */
export class MlService implements MlApi {
  private abort: AbortToken = { aborted: false };
  private predictor: Predictor | null = null;
  private extractor: FeatureExtractor | null = null;
  private customExtractorId: string | null = null;
  private runtime: RuntimeReport | null = null;
  private lastBatchEmit = 0;

  constructor(private readonly emit: Emit, private readonly hostKind: 'worker' | 'inline') {}

  abortTraining(): void {
    this.abort.aborted = true;
  }

  async init(pref: BackendPreference): Promise<RuntimeReport> {
    this.runtime = await initBackend(pref, this.hostKind);
    return this.runtime;
  }

  async memory(): Promise<MemoryReport> {
    return memoryReport();
  }

  async gradientCheck(): Promise<GradientCheckReport> {
    return gradientCheck();
  }

  private async ensureExtractor(id: BackboneId): Promise<{ extractor: FeatureExtractor; ms: number; warning?: string }> {
    const t0 = now();
    if (this.extractor && this.extractor.id === id && this.customExtractorId === null) return { extractor: this.extractor, ms: 0 };
    this.extractor?.dispose();
    this.extractor = null;
    this.customExtractorId = null;
    const r = await loadBackbone(id, (fraction) => this.emit('backbone:progress', { id, fraction }));
    this.extractor = r.extractor;
    return { extractor: r.extractor, ms: now() - t0, warning: r.cacheWriteFailed ? 'The backbone could not be cached in IndexedDB; it will be downloaded again next session.' : undefined };
  }

  /** Custom (user-imported) backbones are re-loaded from the artifacts sent with each request rather
   * than cached across calls, since the worker holds no storage handle of its own — the caller
   * (MlService's consumer) is the one with access to IndexedDB. This keeps the worker's only source
   * of truth for "what artifacts exist" the request itself, never a stale local cache. */
  private async ensureCustomExtractor(id: string, name: string, artifacts: ModelArtifactsData): Promise<{ extractor: FeatureExtractor; ms: number }> {
    const t0 = now();
    if (this.extractor && this.customExtractorId === id) return { extractor: this.extractor, ms: 0 };
    this.extractor?.dispose();
    const extractor = await loadCustomBackbone(id, artifacts);
    this.extractor = extractor;
    this.customExtractorId = id;
    void name;
    return { extractor, ms: now() - t0 };
  }

  async prefetchBackbone(id: BackboneId): Promise<{ cached: boolean; ms: number }> {
    const t0 = now();
    const before = this.extractor?.id === id;
    await this.ensureExtractor(id);
    return { cached: before, ms: now() - t0 };
  }

  async unloadModel(): Promise<void> {
    this.predictor?.dispose();
    this.predictor = null;
  }

  async loadModel(artifacts: ModelArtifactsData, metadata: ModelMetadata, customBackbone?: CustomBackboneInput | null): Promise<{ loadMs: number }> {
    const t0 = now();
    validateArtifacts(artifacts, metadata);
    await this.unloadModel();
    let extractor: FeatureExtractor | null = null;
    if (metadata.customBackbone) {
      if (!customBackbone || customBackbone.id !== metadata.customBackbone.id) throw new Error(`This model needs its custom backbone "${metadata.customBackbone.name}", which was not provided.`);
      extractor = (await this.ensureCustomExtractor(customBackbone.id, customBackbone.name, customBackbone.artifacts)).extractor;
    } else if (metadata.backbone) {
      extractor = (await this.ensureExtractor(metadata.backbone.id)).extractor;
    }
    this.predictor = await Predictor.fromArtifacts(artifacts, metadata, extractor);
    return { loadMs: now() - t0 };
  }

  async predictBitmap(bitmap: ImageBitmap): Promise<PredictionResult> {
    if (!this.predictor) {
      bitmap.close();
      throw new Error('No model is loaded. Load or train a model first.');
    }
    const t0 = now();
    const pixels = bitmapToPixels(bitmap, this.predictor.metadata.preprocessing);
    bitmap.close();
    return this.predictor.predictPixels(pixels, now() - t0);
  }

  async evaluate(samples: RawSample[]): Promise<EvaluationReport> {
    const p = this.predictor;
    if (!p) throw new Error('No model is loaded. Load or train a model first.');
    const spec = p.metadata.preprocessing;
    const idx = new Map(p.metadata.classes.map((c, i) => [c.id, i]));
    const { prepared } = await decodeSamples(samples, idx, spec);
    const size = spec.inputSize * spec.inputSize * 3;
    const C = p.metadata.classes.length;
    const probs = new Float32Array(prepared.length * C);
    for (let i = 0; i < prepared.length; i += 32) {
      const end = Math.min(prepared.length, i + 32);
      const buf = new Uint8Array((end - i) * size);
      for (let k = i; k < end; k++) buf.set((prepared[k] as { pixels: Uint8Array }).pixels, (k - i) * size);
      probs.set(p.predictProbs(buf, end - i), i * C);
      await new Promise<void>((r) => setTimeout(r, 0));
    }
    return buildEvaluationReport({
      classIds: p.metadata.classes.map((c) => c.id),
      classNames: p.metadata.classes.map((c) => c.name),
      sampleIds: prepared.map((s) => s.id),
      yTrue: prepared.map((s) => s.classIndex),
      probs,
    });
  }

  async train(req: TrainRequest): Promise<TrainResponse> {
    const T0 = now();
    const cfg = req.config;
    const timings: PerfTimings = {};
    const warnings: string[] = [];
    this.abort = { aborted: false };
    await this.unloadModel();
    if (!this.runtime) throw new Error('The ML runtime has not been initialised.');
    const caps = { affine: this.runtime.affineAugmentation, affineBatch: this.runtime.affineBatch };
    if (cfg.augmentation.enabled && !caps.affine && (cfg.augmentation.rotationDeg > 0 || cfg.augmentation.zoom > 0 || cfg.augmentation.translate > 0)) {
      warnings.push(`The ${this.runtime.activeBackend} backend cannot run the affine transform kernel, so rotation/zoom/translation were skipped. Flip, brightness and contrast still apply.`);
    }

    const used = req.classes.filter((c) => req.samples.some((s) => s.classId === c.id));
    if (used.length < 2) throw new Error('At least two classes with samples are required.');
    const classIndexById = new Map(used.map((c, i) => [c.id, i]));

    // 0. For transfer learning, load the backbone FIRST: a custom (user-imported) backbone's input
    //    size is only known once it is loaded, and decoding below needs the final size to target.
    let extractor: FeatureExtractor | null = null;
    let customBackboneMeta: { id: string; name: string; featureDim: number } | null = null;
    if (cfg.mode === 'transfer') {
      this.emit('train:phase', { phase: 'backbone', message: cfg.customBackboneId ? `Loading custom backbone "${req.customBackbone?.name ?? cfg.customBackboneId}"` : `Loading ${BACKBONES[cfg.backboneId].label}` });
      if (cfg.customBackboneId) {
        if (!req.customBackbone) throw new Error('The selected custom backbone\'s weights were not provided with this request.');
        const r = await this.ensureCustomExtractor(req.customBackbone.id, req.customBackbone.name, req.customBackbone.artifacts);
        extractor = r.extractor;
        timings.backboneLoadMs = r.ms;
        customBackboneMeta = { id: req.customBackbone.id, name: req.customBackbone.name, featureDim: r.extractor.featureDim };
      } else {
        const r = await this.ensureExtractor(cfg.backboneId);
        extractor = r.extractor;
        timings.backboneLoadMs = r.ms;
        if (r.warning) warnings.push(r.warning);
      }
    }
    const spec: PreprocessingSpec = cfg.mode === 'transfer' ? (extractor as FeatureExtractor).spec : scratchSpec(cfg.scratchInputSize);

    // 1. Decode + preprocess
    this.emit('train:phase', { phase: 'decoding', done: 0, total: req.samples.length });
    let t = now();
    const { prepared, failed } = await decodeSamples(req.samples, classIndexById, spec, (done, total) => this.emit('train:phase', { phase: 'decoding', done, total }), () => this.abort.aborted);
    timings.decodeMs = now() - t;
    if (failed.length) warnings.push(`${failed.length} image(s) could not be decoded and were skipped.`);
    if (this.abort.aborted) throw new Error('Training was cancelled before it started.');

    // 2. Split (seeded, stratified, burst-aware)
    t = now();
    const split = splitDataset(prepared.map((p) => ({ id: p.id, classIndex: p.classIndex, groupId: p.groupId, seq: p.seq })), { validationSplit: cfg.validationSplit, seed: cfg.seed });
    timings.splitMs = now() - t;
    const byId = new Map(prepared.map((p) => [p.id, p]));
    const train = split.train.map((id) => byId.get(id)!);
    const val = split.val.map((id) => byId.get(id)!);
    if (train.length === 0) throw new Error('No training samples remain after the split.');
    if (val.length === 0) warnings.push('The validation set is empty (too few samples per class); metrics reflect training data only.');

    // 3. Build data + model
    const numClasses = used.length;
    let data;
    let model: ClassifierModel;
    if (cfg.mode === 'transfer') {
      const ext = extractor as FeatureExtractor; // guaranteed set above when mode === 'transfer'
      this.emit('train:phase', { phase: 'features', done: 0, total: 1 });
      t = now();
      data = await buildFeatureData(train, val, ext, spec, cfg.augmentation, cfg.augmentedCopies, cfg.seed, caps, numClasses,
        (done, total) => this.emit('train:phase', { phase: 'features', done, total }), () => this.abort.aborted);
      timings.featureExtractionMs = now() - t;
      model = buildHeadModel({ featureDim: ext.featureDim, numClasses, hiddenUnits: cfg.hiddenUnits, dropout: cfg.dropout });
    } else {
      data = buildImageData(train, val, spec, cfg, caps, numClasses);
      model = buildCnn({ inputSize: spec.inputSize, filters: cfg.scratchFilters, numClasses, hiddenUnits: cfg.hiddenUnits, dropout: cfg.dropout });
    }

    // 4. Train
    this.emit('train:phase', { phase: 'training' });
    t = now();
    let engine;
    try {
      engine = await trainClassifier(data, model, {
        epochs: cfg.epochs, batchSize: cfg.batchSize, learningRate: cfg.learningRate, seed: cfg.seed,
        classWeighting: cfg.classWeighting, earlyStoppingPatience: cfg.earlyStoppingPatience, traceEvery: cfg.mode === 'scratch' ? cfg.traceEvery : 0,
      }, {
        onEpochStart: (e) => this.emit('train:epochStart', e),
        onBatchEnd: (m) => {
          const n = now();
          if (n - this.lastBatchEmit > 100 || m.batch === m.batches) {
            this.lastBatchEmit = n;
            this.emit('train:batch', m);
          }
        },
        onEpochEnd: (m) => this.emit('train:epoch', m),
        onValidationEnd: (m) => this.emit('train:validation', m),
        onMechanics: (m) => this.emit('train:mechanics', m),
      }, this.abort);
    } catch (e) {
      data.xTrain.dispose();
      data.xVal?.dispose();
      model.exportModel.dispose();
      throw e;
    }
    timings.trainMs = now() - t;

    // 5. Evaluate on the held-out split (best weights already restored)
    this.emit('train:phase', { phase: 'evaluating' });
    t = now();
    let evaluation: EvaluationReport | null = null;
    if (data.xVal && data.yVal) {
      const ev = await evaluateSplit(model.trainModel, data.xVal, data.yVal, numClasses, cfg.batchSize, data.valTransform);
      evaluation = buildEvaluationReport({ classIds: used.map((c) => c.id), classNames: used.map((c) => c.name), sampleIds: split.val, yTrue: data.yVal, probs: ev.probs });
    }
    timings.evaluateMs = now() - t;
    data.xTrain.dispose();
    data.xVal?.dispose();

    // 6. Package
    this.emit('train:phase', { phase: 'finalizing' });
    const artifacts = await extractArtifacts(model.exportModel);
    const best = engine.history[Math.max(0, (engine.bestEpoch || engine.history.length) - 1)];
    const inputShape: [number, number, number] = [spec.inputSize, spec.inputSize, 3];
    const metadata: ModelMetadata = {
      schemaVersion: 1,
      modelType: cfg.mode === 'transfer' ? 'transfer-mobilenetv2' : 'scratch-cnn',
      classes: used.map((c) => ({ id: c.id, name: c.name })),
      inputShape,
      modelInputShape: cfg.mode === 'transfer' ? [(extractor as FeatureExtractor).featureDim] : inputShape,
      preprocessing: spec,
      backbone: cfg.mode === 'transfer' && !cfg.customBackboneId ? backboneRef(cfg.backboneId) : null,
      customBackbone: cfg.mode === 'transfer' && customBackboneMeta ? customBackboneMeta : null,
      outputActivation: 'softmax',
      trainingConfig: cfg,
      finalMetrics: {
        loss: best?.loss ?? null, accuracy: best?.accuracy ?? null, valLoss: best?.valLoss ?? null,
        valAccuracy: evaluation?.accuracy ?? best?.valAccuracy ?? null, macroF1: evaluation?.macroF1 ?? null,
      },
      createdAt: new Date().toISOString(),
      appVersion: APP_VERSION,
    };
    // Keep the trained model resident for immediate inference (train/export models share layers and weights).
    this.predictor = new Predictor(model.exportModel, metadata, extractor, true);
    timings.totalMs = now() - T0;

    return {
      metadata, artifacts, evaluation, history: engine.history,
      trainIds: split.train, valIds: split.val, purgedIds: split.purged, failedSampleIds: failed,
      durationMs: timings.totalMs, paramCount: engine.paramCount, modelBytes: artifacts.weightData.byteLength, timings,
      backend: this.runtime.activeBackend, stoppedEarly: engine.stoppedEarly, aborted: engine.aborted, warnings,
    };
  }
}
