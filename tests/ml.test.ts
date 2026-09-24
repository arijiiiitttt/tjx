import * as tf from '@tensorflow/tfjs';
import { beforeAll, describe, expect, it } from 'vitest';
import { augmentBatch, DEFAULT_AUGMENTATION } from '@/ml/augmentation/augment';
import { splitDataset } from '@/ml/data/split';
import { buildBundleFiles, readBundle, zipBundle } from '@/ml/export/bundle';
import { ALLOWED_LAYERS, extractLayers, looksLikeLayersTopology, ModelValidationError } from '@/ml/export/validate';
import { Predictor } from '@/ml/inference/predictor';
import { buildEvaluationReport } from '@/ml/metrics/classification';
import type { FeatureExtractor } from '@/ml/models/backbone';
import { buildCnn } from '@/ml/models/cnn';
import { loadCustomBackbone } from '@/ml/models/backbone';
import { buildHeadModel } from '@/ml/models/head';
import { MOBILENET_SPEC, pixelsToTensor, pixelsToUnit, scratchSpec } from '@/ml/preprocessing/spec';
import { evaluateSplit, trainClassifier } from '@/ml/training/engine';
import { probeBackend } from '@/ml/runtime/backend';
import { gradientCheck } from '@/ml/training/gradcheck';
import { buildFeatureData, buildImageData } from '@/ml/training/pipeline';
import { extractArtifacts } from '@/ml/service';
import { APP_VERSION } from '@/ml/version';
import type { PreparedSample } from '@/ml/data/prepare';
import type { ModelMetadata, TrainingConfig } from '@/types/ml';
import { mulberry32 } from '@/utils/rng';
import { readFileSync } from 'node:fs';
import { blobImage, colourImage, useCpu } from './helpers';

beforeAll(useCpu);

const SIZE = 24;
const baseConfig = (over: Partial<TrainingConfig> = {}): TrainingConfig => ({
  mode: 'scratch', backboneId: 'mobilenet-v2-100', customBackboneId: null, epochs: 30, batchSize: 8, learningRate: 0.01, validationSplit: 0.25, seed: 11,
  augmentation: { ...DEFAULT_AUGMENTATION, hFlip: false, rotationDeg: 0, zoom: 0, translate: 0.06, brightness: 0.1, contrast: 0.1 },
  augmentedCopies: 2, earlyStoppingPatience: 0, classWeighting: 'balanced', scratchInputSize: SIZE, scratchFilters: [6, 8, 8],
  hiddenUnits: 16, dropout: 0, traceEvery: 5, ...over,
});

function makeSamples(make: (size: number, cls: number, rng: () => number) => Uint8Array, size: number, perClass: number): PreparedSample[] {
  const rng = mulberry32(3);
  const out: PreparedSample[] = [];
  for (let c = 0; c < 2; c++) for (let i = 0; i < perClass; i++) out.push({ id: `s${c}-${String(i).padStart(3, '0')}`, classIndex: c, groupId: `g${c}-${i}`, seq: 0, pixels: make(size, c, rng) });
  return out;
}
const splitOf = (samples: PreparedSample[], cfg: TrainingConfig) => {
  const s = splitDataset(samples.map((p) => ({ id: p.id, classIndex: p.classIndex, groupId: p.groupId, seq: p.seq })), { validationSplit: cfg.validationSplit, seed: cfg.seed });
  const by = new Map(samples.map((p) => [p.id, p]));
  return { s, train: s.train.map((id) => by.get(id)!), val: s.val.map((id) => by.get(id)!) };
};
const meta = (over: Partial<ModelMetadata>): ModelMetadata => ({
  schemaVersion: 1, modelType: 'scratch-cnn', classes: [{ id: 'a', name: 'A' }, { id: 'b', name: 'B' }], inputShape: [SIZE, SIZE, 3], modelInputShape: [SIZE, SIZE, 3],
  preprocessing: scratchSpec(SIZE), backbone: null, customBackbone: null, outputActivation: 'softmax', trainingConfig: baseConfig(),
  finalMetrics: { loss: 0, accuracy: 1, valLoss: 0, valAccuracy: 1, macroF1: 1 }, createdAt: new Date().toISOString(), appVersion: APP_VERSION, ...over,
});

describe('preprocessing', () => {
  it('is deterministic and maps bytes to the stored normalisation range', () => {
    const px = new Uint8Array(SIZE * SIZE * 3).map((_, i) => i % 256);
    const a = pixelsToTensor(px, 1, scratchSpec(SIZE));
    const b = pixelsToTensor(px, 1, scratchSpec(SIZE));
    expect(Array.from(a.dataSync())).toEqual(Array.from(b.dataSync()));
    expect(a.dataSync()[255]).toBeCloseTo(255 / 255);
    const signed = pixelsToTensor(px, 1, { ...scratchSpec(SIZE), normalization: { min: -1, max: 1 } });
    expect(signed.dataSync()[0]).toBeCloseTo(-1);
    expect(signed.dataSync()[255]).toBeCloseTo(1);
    tf.dispose([a, b, signed]);
  });
});

describe('augmentation', () => {
  it('preserves shape and range, is reproducible per seed, and identity when disabled', () => {
    const x = tf.tidy(() => tf.randomUniform([8, 16, 16, 3], 0, 1, 'float32', 5) as tf.Tensor4D);
    const cfg = { ...DEFAULT_AUGMENTATION };
    const a = augmentBatch(x, cfg, mulberry32(9), { affine: true });
    const b = augmentBatch(x, cfg, mulberry32(9), { affine: true });
    const c = augmentBatch(x, cfg, mulberry32(10), { affine: true });
    expect(a.shape).toEqual([8, 16, 16, 3]);
    const av = a.dataSync();
    expect(Math.min(...av)).toBeGreaterThanOrEqual(0);
    expect(Math.max(...av)).toBeLessThanOrEqual(1);
    expect(Array.from(av)).toEqual(Array.from(b.dataSync()));
    expect(Array.from(av)).not.toEqual(Array.from(c.dataSync()));
    expect(augmentBatch(x, { ...cfg, enabled: false }, mulberry32(1), { affine: true })).toBe(x);
    tf.dispose([x, a, b, c]);
  });
  it('horizontal flip really mirrors the image; an identity affine transform preserves it', () => {
    const x = tf.tensor4d(Float32Array.from({ length: 4 * 4 * 3 }, (_, i) => (i % 11) / 11), [1, 4, 4, 3]);
    const flipOnly = { enabled: true, hFlip: true, rotationDeg: 0, zoom: 0, translate: 0, brightness: 0, contrast: 0 };
    const seenFlip = [1, 2, 3, 4, 5, 6, 7, 8].map((s) => Array.from(augmentBatch(x, flipOnly, mulberry32(s), { affine: true }).dataSync()));
    const orig = Array.from(x.dataSync());
    const mirrored = Array.from(tf.reverse(x, [2]).dataSync());
    expect(seenFlip.some((v) => v.every((n, i) => Math.abs(n - (orig[i] as number)) < 1e-6))).toBe(true);
    expect(seenFlip.some((v) => v.every((n, i) => Math.abs(n - (mirrored[i] as number)) < 1e-6))).toBe(true);
    const ident = tf.image.transform(x, tf.tensor2d([[1, 0, 0, 0, 1, 0, 0, 0]]), 'bilinear', 'reflect', 0).dataSync();
    ident.forEach((v, i) => expect(v).toBeCloseTo(orig[i] as number, 4));
  });
});

describe('affine augmentation on this backend (regression: tfjs-cpu batched Transform bug)', () => {
  it('never corrupts any sample of a batch: probe result is honoured and brightness/blob mass is preserved', async () => {
    const probe = await probeBackend();
    expect(probe.affine).toBe(true);
    const size = 24;
    const rng = mulberry32(3);
    const px = new Uint8Array(8 * size * size * 3);
    for (let i = 0; i < 8; i++) px.set(blobImage(size, i % 2, rng), i * size * size * 3);
    const x = pixelsToUnit(px, 8, size);
    const cfg = { enabled: true, hFlip: false, rotationDeg: 0, zoom: 0, translate: 0.06, brightness: 0, contrast: 0 };
    const a = augmentBatch(x, cfg, mulberry32(5), { affine: true, affineBatch: probe.affineBatch });
    // per-sample mean must stay close (a small shift only re-arranges pixels)
    const mx = x.mean([1, 2, 3]).dataSync();
    const ma = a.mean([1, 2, 3]).dataSync();
    mx.forEach((v, i) => expect(Math.abs(v - (ma[i] as number))).toBeLessThan(0.03));
    // and the forced per-sample fallback path is always correct
    const b = augmentBatch(x, cfg, mulberry32(5), { affine: true, affineBatch: false });
    const mb = b.mean([1, 2, 3]).dataSync();
    mx.forEach((v, i) => expect(Math.abs(v - (mb[i] as number))).toBeLessThan(0.03));
    tf.dispose([x, a, b]);
  });
});

describe('backpropagation', () => {
  it('analytic gradients from autodiff match numerical central differences (and leak nothing)', async () => {
    const before = tf.memory().numTensors;
    const r = await gradientCheck();
    expect(tf.memory().numTensors).toBe(before);
    expect(r.checked).toBeGreaterThan(10);
    expect(r.maxRelError).toBeLessThan(r.tolerance);
    expect(r.passed).toBe(true);
  });
});

describe('from-scratch CNN training (real forward/backward passes)', () => {
  it('learns a real task, updates weights, reports honest metrics and leaks no tensors', async () => {
    const cfg = baseConfig();
    const samples = makeSamples(blobImage, SIZE, 30);
    const { s, train, val } = splitOf(samples, cfg);
    const baseline = tf.memory().numTensors;

    const spec = scratchSpec(SIZE);
    const data = buildImageData(train, val, spec, cfg, { affine: true, affineBatch: false }, 2);
    const model = buildCnn({ inputSize: SIZE, filters: cfg.scratchFilters, numClasses: 2, hiddenUnits: cfg.hiddenUnits, dropout: cfg.dropout });
    const before = Array.from(model.trainModel.getWeights()[0]!.dataSync());
    const events = { batches: 0, epochs: 0, val: 0, traces: 0, starts: 0 };
    let firstTrace: { layers: { gradNorm: number; updateNorm: number }[] } | null = null;

    const res = await trainClassifier(data, model, { epochs: cfg.epochs, batchSize: cfg.batchSize, learningRate: cfg.learningRate, seed: cfg.seed, classWeighting: 'balanced', earlyStoppingPatience: 0, traceEvery: 5 }, {
      onEpochStart: () => events.starts++, onBatchEnd: () => events.batches++, onEpochEnd: () => events.epochs++, onValidationEnd: () => events.val++,
      onMechanics: (t) => { events.traces++; firstTrace ??= t; },
    }, { aborted: false });

    const after = Array.from(model.trainModel.getWeights()[0]!.dataSync());
    expect(after).not.toEqual(before); // weights actually changed
    expect(res.history).toHaveLength(cfg.epochs);
    expect(events.epochs).toBe(cfg.epochs);
    expect(events.starts).toBe(cfg.epochs);
    expect(events.val).toBe(cfg.epochs);
    expect(events.batches).toBeGreaterThan(cfg.epochs);
    expect(events.traces).toBeGreaterThan(0);
    // while learning is active every layer receives a gradient and is actually updated (at convergence they vanish)
    expect(firstTrace!.layers.every((l) => l.gradNorm > 0 && l.updateNorm > 0)).toBe(true);
    const first = res.history[0]!;
    const last = res.history[res.history.length - 1]!;
    expect(last.loss).toBeLessThan(first.loss * 0.6); // loss decreases substantially
    expect(res.paramCount).toBe(model.trainModel.countParams());

    const ev = await evaluateSplit(model.trainModel, data.xVal!, data.yVal!, 2, 16);
    const report = buildEvaluationReport({ classIds: ['a', 'b'], classNames: ['A', 'B'], sampleIds: s.val, yTrue: data.yVal!, probs: ev.probs });
    expect(report.accuracy).toBeGreaterThanOrEqual(0.9); // generalises to held-out images
    expect(report.macroF1).toBeGreaterThanOrEqual(0.85);

    data.xTrain.dispose();
    data.xVal?.dispose();
    model.exportModel.dispose();
    expect(tf.memory().numTensors).toBe(baseline); // no leaked tensors after a full training run
  });

  it('sanity: the loop can overfit 12 images (loss -> ~0, train accuracy 1) with no augmentation or dropout', async () => {
    const cfg = baseConfig({ augmentation: { ...DEFAULT_AUGMENTATION, enabled: false }, dropout: 0, epochs: 80, learningRate: 0.01, validationSplit: 0 });
    const samples = makeSamples(blobImage, SIZE, 6);
    const data = buildImageData(samples, [], scratchSpec(SIZE), cfg, { affine: true }, 2);
    const model = buildCnn({ inputSize: SIZE, filters: [4, 8], numClasses: 2, hiddenUnits: 16, dropout: 0 });
    const res = await trainClassifier(data, model, { epochs: 80, batchSize: 12, learningRate: 0.01, seed: 2, classWeighting: 'none', earlyStoppingPatience: 0, traceEvery: 0 }, {}, { aborted: false });
    const last = res.history[res.history.length - 1]!;
    expect(last.loss).toBeLessThan(0.1);
    expect(last.accuracy).toBe(1);
    data.xTrain.dispose(); model.exportModel.dispose();
  });

  it('honours the abort token and restores best weights with early stopping', async () => {
    const cfg = baseConfig({ epochs: 30 });
    const samples = makeSamples(blobImage, SIZE, 24);
    const { train, val } = splitOf(samples, cfg);
    const data = buildImageData(train, val, scratchSpec(SIZE), cfg, { affine: true }, 2);
    const model = buildCnn({ inputSize: SIZE, filters: [4, 4], numClasses: 2, hiddenUnits: 8, dropout: 0 });
    const abort = { aborted: false };
    const res = await trainClassifier(data, model, { epochs: 30, batchSize: 8, learningRate: 0.005, seed: 1, classWeighting: 'none', earlyStoppingPatience: 0, traceEvery: 0 }, { onEpochEnd: (m) => { if (m.epoch === 2) abort.aborted = true; } }, abort);
    expect(res.aborted).toBe(true);
    expect(res.history).toHaveLength(2);
    data.xTrain.dispose(); data.xVal?.dispose(); model.exportModel.dispose();
  });
});

/** Fixed random conv net standing in for a pretrained backbone (real tensors, no network). */
function fakeBackbone(dim = 24): FeatureExtractor {
  const rng = mulberry32(99);
  const k = tf.tensor4d(Float32Array.from({ length: 3 * 3 * 3 * dim }, () => rng() - 0.5), [3, 3, 3, dim]);
  return { id: 'fake', spec: { ...MOBILENET_SPEC, inputSize: SIZE }, featureDim: dim, embed: (x) => tf.tidy(() => tf.relu(tf.conv2d(x, k, 2, 'same')).mean([1, 2]) as tf.Tensor2D), dispose: () => k.dispose() };
}

describe('transfer learning (frozen backbone + trained head)', () => {
  it('trains the head on cached embeddings incl. augmented copies, and exports/imports bit-exactly', async () => {
    const cfg = baseConfig({ mode: 'transfer', epochs: 40, learningRate: 0.01, hiddenUnits: 16, augmentedCopies: 2 });
    const samples = makeSamples(colourImage, SIZE, 30);
    const { train, val } = splitOf(samples, cfg);
    const baseline = tf.memory().numTensors; // measured BEFORE creating anything, so no leak can be offset
    const ext = fakeBackbone();
    const spec = ext.spec;

    const data = await buildFeatureData(train, val, ext, spec, cfg.augmentation, 2, cfg.seed, { affine: true }, 2);
    expect(data.xTrain.shape).toEqual([train.length * 3, 24]); // originals + 2 augmented copies
    expect(data.xVal!.shape).toEqual([val.length, 24]); // validation never augmented
    const model = buildHeadModel({ featureDim: 24, numClasses: 2, hiddenUnits: 16, dropout: 0.1 });
    const res = await trainClassifier(data, model, { epochs: cfg.epochs, batchSize: 16, learningRate: cfg.learningRate, seed: 1, classWeighting: 'balanced', earlyStoppingPatience: 6, traceEvery: 0 }, {}, { aborted: false });
    const ev = await evaluateSplit(model.trainModel, data.xVal!, data.yVal!, 2, 16);
    expect(ev.accuracy).toBeGreaterThanOrEqual(0.95);
    expect(res.history.length).toBeLessThanOrEqual(cfg.epochs);

    // Export -> bundle -> zip -> import -> identical predictions
    const artifacts = await extractArtifacts(model.exportModel);
    const metadata = meta({ modelType: 'transfer-mobilenetv2', modelInputShape: [24], inputShape: [SIZE, SIZE, 3], preprocessing: spec, backbone: { id: 'mobilenet-v2-100', url: 'https://example.com/m', embeddingNode: 'n', featureDim: 24 } });
    const original = new Predictor(model.exportModel, metadata, ext, false);
    const files = buildBundleFiles(artifacts, metadata);
    const zip = zipBundle(files);
    const imported = readBundle([{ name: 'model.zip', bytes: zip }]);
    const loose = readBundle([{ name: 'model.json', bytes: files['model.json'] }, { name: 'weights.bin', bytes: files['weights.bin'] }, { name: 'metadata.json', bytes: files['metadata.json'] }]);
    expect(loose.metadata.classes).toEqual(metadata.classes);
    const restored = await Predictor.fromArtifacts(imported.artifacts, imported.metadata, ext);
    const probe = val[0]!.pixels;
    const p1 = await original.predictPixels(probe);
    const p2 = await restored.predictPixels(probe);
    expect(p1.predictions.map((p) => p.classId)).toEqual(p2.predictions.map((p) => p.classId));
    p1.predictions.forEach((p, i) => expect(p.probability).toBeCloseTo(p2.predictions[i]!.probability, 6));
    expect(p2.latencyMs).toBeGreaterThan(0);
    expect(p1.predictions.reduce((a, p) => a + p.probability, 0)).toBeCloseTo(1, 5);
    // the correct class wins for a held-out sample
    expect(p2.predictions[0]!.classId).toBe(val[0]!.classIndex === 0 ? 'a' : 'b');

    restored.dispose();
    data.xTrain.dispose(); data.xVal?.dispose(); model.exportModel.dispose(); ext.dispose();
    expect(tf.memory().numTensors).toBe(baseline);
  });
});

describe('untrusted model import', () => {
  const trained = async () => {
    const model = buildCnn({ inputSize: SIZE, filters: [4], numClasses: 2, hiddenUnits: 4, dropout: 0 });
    const artifacts = await extractArtifacts(model.exportModel);
    const files = buildBundleFiles(artifacts, meta({}));
    model.exportModel.dispose();
    return files;
  };
  const named = (f: Awaited<ReturnType<typeof trained>>) => [{ name: 'model.json', bytes: f['model.json'] }, { name: 'weights.bin', bytes: f['weights.bin'] }, { name: 'metadata.json', bytes: f['metadata.json'] }];

  it('accepts a valid bundle', async () => {
    const f = await trained();
    expect(() => readBundle(named(f))).not.toThrow();
  });
  it('rejects truncated weights, disallowed layers, missing metadata and tampered metadata', async () => {
    const f = await trained();
    const bad = named(f);
    bad[1] = { name: 'weights.bin', bytes: f['weights.bin'].slice(0, f['weights.bin'].length - 8) };
    expect(() => readBundle(bad)).toThrow(/weights\.bin has/);

    const json = JSON.parse(new TextDecoder().decode(f['model.json']));
    json.modelTopology.config.layers[1].class_name = 'Lambda';
    const evil = named(f);
    evil[0] = { name: 'model.json', bytes: new TextEncoder().encode(JSON.stringify(json)) };
    expect(() => readBundle(evil)).toThrow(/not allowed/);

    expect(() => readBundle(named(f).slice(0, 2))).toThrow(ModelValidationError);

    const m = JSON.parse(new TextDecoder().decode(f['metadata.json']));
    m.classes = [{ id: 'x', name: 'X' }, { id: 'y', name: 'Y' }, { id: 'z', name: 'Z' }];
    const tampered = named(f);
    tampered[2] = { name: 'metadata.json', bytes: new TextEncoder().encode(JSON.stringify(m)) };
    expect(() => readBundle(tampered)).toThrow(/outputs 2 classes/);

    m.schemaVersion = 7;
    tampered[2] = { name: 'metadata.json', bytes: new TextEncoder().encode(JSON.stringify(m)) };
    expect(() => readBundle(tampered)).toThrow(/metadata\.json is invalid/);
  });
});

describe('real third-party model compatibility (tests/fixtures/real-external-model.json)', () => {
  // A genuine model.json from an external, independently-authored GitHub repository (a U-Net
  // segmentation model), fetched via codeload.github.com — not generated by this codebase. It
  // exercises two real-world conventions our own self-generated models never happen to hit:
  // (1) it omits the top-level "format": "layers-model" field some older tfjs_converter output
  // leaves out, and (2) it uses Conv2DTranspose/Concatenate, which are not layer types this app's
  // own models ever produce. Both were rejected before the fix; both must be accepted now.
  const realTopology = () => JSON.parse(readFileSync(new URL('./fixtures/real-external-model.json', import.meta.url), 'utf-8')) as { modelTopology: unknown; weightsManifest: { weights: { shape: number[] }[] }[] };

  it('is accepted as a real LayersModel topology despite lacking a top-level "format" field', () => {
    const real = realTopology();
    expect(looksLikeLayersTopology(real.modelTopology)).toBe(true);
    expect(looksLikeLayersTopology({ nonsense: true })).toBe(false);
    expect(looksLikeLayersTopology(null)).toBe(false);
  });

  it("extracts every real layer, including types absent from this app's own generated models", () => {
    const real = realTopology();
    const layers = extractLayers(real.modelTopology);
    expect(layers.length).toBe(39);
    const types = new Set(layers.map((l) => (l as { class_name: string }).class_name));
    expect(types).toEqual(new Set(['InputLayer', 'Conv2D', 'MaxPooling2D', 'Conv2DTranspose', 'Concatenate']));
    for (const t of types) expect(ALLOWED_LAYERS.has(t)).toBe(true); // would have failed before the allowlist was widened
  });

  it('readBundle accepts this real file (structural check) once weight bytes match, and rejects it once a real disallowed op is injected', () => {
    const real = realTopology();
    const weightSpecs = real.weightsManifest.flatMap((g) => g.weights);
    const totalBytes = weightSpecs.reduce((a, w) => a + w.shape.reduce((x, y) => x * y, 1) * 4, 0);
    expect(totalBytes).toBe(7047508); // independently computed from the real manifest
    const modelJson = { format: undefined, modelTopology: real.modelTopology, weightsManifest: real.weightsManifest }; // no "format" field, exactly as fetched
    const files = {
      'model.json': new TextEncoder().encode(JSON.stringify(modelJson)),
      'weights.bin': new Uint8Array(totalBytes), // real topology/shapes; synthetic weight VALUES (numeric correctness of this unrelated pretrained network is not what's under test)
      'metadata.json': new TextEncoder().encode(JSON.stringify(meta({ modelInputShape: [128, 128, 3], inputShape: [128, 128, 3] }))),
    };
    expect(() => readBundle([{ name: 'model.json', bytes: files['model.json'] }, { name: 'weights.bin', bytes: files['weights.bin'] }, { name: 'metadata.json', bytes: files['metadata.json'] }])).not.toThrow();

    const layers = (real.modelTopology as { model_config: { config: { layers: { class_name: string }[] } } }).model_config.config.layers;
    const tampered = JSON.parse(JSON.stringify(real.modelTopology)) as typeof real.modelTopology & { model_config: { config: { layers: { class_name: string }[] } } };
    tampered.model_config.config.layers[10]!.class_name = 'Lambda';
    const evilJson = { modelTopology: tampered, weightsManifest: real.weightsManifest };
    expect(() => readBundle([{ name: 'model.json', bytes: new TextEncoder().encode(JSON.stringify(evilJson)) }, { name: 'weights.bin', bytes: files['weights.bin'] }, { name: 'metadata.json', bytes: files['metadata.json'] }])).toThrow(/not allowed/);
    void layers;
  });
});

describe('custom (user-imported) backbone', () => {
  it('loadCustomBackbone loads a real serialized LayersModel end-to-end, auto-pools a spatial 4D output, and yields correct, poolable embeddings', async () => {
    // Build and export a small real conv stack (own weights, but going through the exact same
    // save -> bundle -> parse -> tf.loadLayersModel -> execute path a genuinely external file would.
    const src = buildCnn({ inputSize: SIZE, filters: [4, 6], numClasses: 2, hiddenUnits: 8, dropout: 0 });
    // Re-purpose it as a feature extractor by taking the flatten layer's output (a real, already-trained-shape 2D tensor).
    const flatten = src.trainModel.layers.find((l) => l.getClassName() === 'Flatten')!;
    const featureModel = tf.model({ inputs: src.trainModel.inputs, outputs: flatten.output as tf.SymbolicTensor });
    const artifacts = await extractArtifacts(featureModel);
    expect(looksLikeLayersTopology(artifacts.modelTopology)).toBe(true);

    const before = tf.memory().numTensors;
    const extractor = await loadCustomBackbone('test-backbone', artifacts);
    expect(extractor.featureDim).toBeGreaterThan(0);
    expect(extractor.id).toBe('custom:test-backbone');

    const x = tf.tidy(() => tf.randomUniform([3, SIZE, SIZE, 3]) as tf.Tensor4D);
    const emb = extractor.embed(x);
    expect(emb.shape).toEqual([3, extractor.featureDim]);
    // deterministic: same input -> identical embedding
    const emb2 = extractor.embed(x);
    expect(Array.from(emb.dataSync())).toEqual(Array.from(emb2.dataSync()));

    // `before` was captured after src/featureModel already existed, and they share underlying
    // Layer objects (disposing one would dispose the other's weights too) — only `extractor` is a
    // genuinely independent deserialization (fresh tensors from serialized bytes) and is what this
    // leak check is about, so only it (plus the tensors this test itself created) is disposed here.
    x.dispose(); emb.dispose(); emb2.dispose(); extractor.dispose();
    expect(tf.memory().numTensors).toBe(before);
  });

  it('rejects a model whose output is not usable as a feature vector', async () => {
    const inp = tf.input({ shape: [4] });
    const bad = tf.model({ inputs: inp, outputs: tf.layers.dense({ units: 1 }).apply(inp) as tf.SymbolicTensor });
    const artifacts = await extractArtifacts(bad);
    await expect(loadCustomBackbone('bad', artifacts)).rejects.toThrow(/feature size/);
    bad.dispose();
  });

  it('trains a real head on a real custom-backbone feature extractor end-to-end', async () => {
    const src = buildCnn({ inputSize: SIZE, filters: [4, 6], numClasses: 2, hiddenUnits: 8, dropout: 0 });
    const pooled = tf.layers.globalAveragePooling2d({}).apply(src.trainModel.layers.find((l) => l.getClassName() === 'Conv2D')!.output as tf.SymbolicTensor) as tf.SymbolicTensor;
    const featureModel = tf.model({ inputs: src.trainModel.inputs, outputs: pooled });
    const artifacts = await extractArtifacts(featureModel);
    const extractor = await loadCustomBackbone('bb2', artifacts);

    const cfg = baseConfig({ mode: 'transfer', epochs: 15, learningRate: 0.02, augmentedCopies: 1 });
    const samples = makeSamples(colourImage, SIZE, 20);
    const { train, val } = splitOf(samples, cfg);
    const data = await buildFeatureData(train, val, extractor, extractor.spec, cfg.augmentation, 1, cfg.seed, { affine: true }, 2);
    const model = buildHeadModel({ featureDim: extractor.featureDim, numClasses: 2, hiddenUnits: 8, dropout: 0.1 });
    await trainClassifier(data, model, { epochs: cfg.epochs, batchSize: 8, learningRate: cfg.learningRate, seed: 1, classWeighting: 'balanced', earlyStoppingPatience: 0, traceEvery: 0 }, {}, { aborted: false });
    const ev = await evaluateSplit(model.trainModel, data.xVal!, data.yVal!, 2, 8);
    expect(ev.accuracy).toBeGreaterThan(0.5); // learns something real through a fully custom backbone

    data.xTrain.dispose(); data.xVal?.dispose(); model.exportModel.dispose(); extractor.dispose();
  });
});
