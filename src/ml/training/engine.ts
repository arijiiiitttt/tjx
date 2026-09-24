import * as tf from '@tensorflow/tfjs';
import type { BatchMetrics, EpochMetrics, LayerTrace, MechanicsTrace } from '@/types/ml';
import { mulberry32, shuffleInPlace } from '@/utils/rng';
import type { ClassifierModel } from '../models/head';

/** A device-resident dataset. `x` rows are model inputs (images or cached embeddings). */
export interface TrainData {
  xTrain: tf.Tensor;
  yTrain: Int32Array;
  xVal: tf.Tensor | null;
  yVal: Int32Array | null;
  numClasses: number;
  /** Applied per batch inside the gradient scope (fresh random augmentation + normalisation). */
  batchAugment?: (x: tf.Tensor, step: number) => tf.Tensor;
  /** Applied to validation batches (normalisation only). */
  valTransform?: (x: tf.Tensor) => tf.Tensor;
}

export interface EngineOptions {
  epochs: number;
  batchSize: number;
  learningRate: number;
  seed: number;
  classWeighting: 'none' | 'balanced';
  earlyStoppingPatience: number;
  traceEvery: number;
}

export interface TrainingHooks {
  onEpochStart?(e: { epoch: number; epochs: number }): void;
  onBatchEnd?(m: BatchMetrics): void;
  onEpochEnd?(m: EpochMetrics): void;
  onValidationEnd?(m: { epoch: number; valLoss: number; valAccuracy: number }): void;
  onMechanics?(t: MechanicsTrace): void;
  onTrainingComplete?(r: EngineResult): void;
  onTrainingError?(err: Error): void;
}

export interface AbortToken {
  aborted: boolean;
}

export interface EngineResult {
  history: EpochMetrics[];
  bestEpoch: number;
  stoppedEarly: boolean;
  aborted: boolean;
  durationMs: number;
  paramCount: number;
  steps: number;
}

export interface SplitEval {
  loss: number;
  accuracy: number;
  preds: Int32Array;
  /** flattened [n * numClasses] softmax probabilities */
  probs: Float32Array;
}

/** Class-weighted softmax cross-entropy computed from logits: L = sum_i w_i * (-log softmax(z_i)[y_i]) / sum_i w_i */
export function weightedSoftmaxCE(logits: tf.Tensor, onehot: tf.Tensor, w: tf.Tensor): tf.Scalar {
  const logp = tf.logSoftmax(logits);
  const perSample = tf.neg(tf.sum(tf.mul(onehot, logp), 1));
  return tf.div(tf.sum(tf.mul(perSample, w)), tf.sum(w)) as tf.Scalar;
}

export function classWeightVector(labels: Int32Array, numClasses: number, mode: 'none' | 'balanced'): Float32Array {
  const w = new Float32Array(numClasses).fill(1);
  if (mode === 'none') return w;
  const counts = new Array<number>(numClasses).fill(0);
  labels.forEach((l) => (counts[l] = (counts[l] as number) + 1));
  for (let c = 0; c < numClasses; c++) w[c] = counts[c] ? labels.length / (numClasses * (counts[c] as number)) : 0;
  return w;
}

const now = () => performance.now();

/** Evaluate in inference mode (dropout off). Never mutates weights. */
export async function evaluateSplit(model: tf.LayersModel, x: tf.Tensor, labels: Int32Array, numClasses: number, batchSize: number, transform?: (x: tf.Tensor) => tf.Tensor): Promise<SplitEval> {
  const n = labels.length;
  const probs = new Float32Array(n * numClasses);
  const preds = new Int32Array(n);
  let lossSum = 0;
  let correct = 0;
  for (let i = 0; i < n; i += batchSize) {
    const end = Math.min(n, i + batchSize);
    const idx = Array.from({ length: end - i }, (_, k) => i + k);
    const { p, l } = tf.tidy(() => {
      const xb0 = tf.gather(x, tf.tensor1d(idx, 'int32'));
      const xb = transform ? transform(xb0) : xb0;
      const logits = model.predict(xb) as tf.Tensor;
      const oh = tf.oneHot(tf.tensor1d(Array.from(labels.subarray(i, end)), 'int32'), numClasses);
      const per = tf.neg(tf.sum(tf.mul(oh, tf.logSoftmax(logits)), 1));
      return { p: tf.softmax(logits), l: per };
    });
    const [pd, ld] = await Promise.all([p.data() as Promise<Float32Array>, l.data() as Promise<Float32Array>]);
    p.dispose();
    l.dispose();
    for (let k = 0; k < end - i; k++) {
      let best = 0;
      let bv = -1;
      for (let c = 0; c < numClasses; c++) {
        const v = pd[k * numClasses + c] as number;
        probs[(i + k) * numClasses + c] = v;
        if (v > bv) {
          bv = v;
          best = c;
        }
      }
      preds[i + k] = best;
      if (best === labels[i + k]) correct++;
      lossSum += ld[k] as number;
    }
  }
  return { loss: n ? lossSum / n : 0, accuracy: n ? correct / n : 0, preds, probs };
}

const yieldToEventLoop = () => new Promise<void>((r) => setTimeout(r, 0));

/**
 * The training loop, written out explicitly:
 *   forward pass -> loss -> reverse-mode gradients (backpropagation) -> Adam weight update.
 * `tf.variableGrads` records every op executed by the forward pass and replays each op's gradient
 * function in reverse; nothing here is simulated.
 */
export async function trainClassifier(data: TrainData, model: ClassifierModel, opts: EngineOptions, hooks: TrainingHooks, abort: AbortToken): Promise<EngineResult> {
  const t0 = now();
  const n = data.yTrain.length;
  const C = data.numClasses;
  if (n === 0) throw new Error('The training set is empty.');

  const rng = mulberry32(opts.seed);
  const order = Array.from({ length: n }, (_, i) => i);
  const optimizer = tf.train.adam(opts.learningRate);
  const vars = model.trainModel.trainableWeights.map((w) => w.read() as tf.Variable);
  const paramCount = model.trainModel.countParams();
  const cw = classWeightVector(data.yTrain, C, opts.classWeighting);
  const yOneHot = tf.tidy(() => tf.oneHot(tf.tensor1d(Array.from(data.yTrain), 'int32'), C));
  const sampleW = tf.tensor1d(Array.from(data.yTrain, (l) => cw[l] as number));

  const history: EpochMetrics[] = [];
  let bestValLoss = Infinity;
  let bestEpoch = 0;
  let bestWeights: tf.Tensor[] | null = null;
  let sinceBest = 0;
  let stoppedEarly = false;
  let step = 0;
  const batches = Math.ceil(n / opts.batchSize);

  const disposeBest = () => {
    bestWeights?.forEach((w) => w.dispose());
    bestWeights = null;
  };

  try {
    for (let epoch = 1; epoch <= opts.epochs && !abort.aborted; epoch++) {
      hooks.onEpochStart?.({ epoch, epochs: opts.epochs });
      const et0 = now();
      shuffleInPlace(order, rng);
      let lossSum = 0;
      let accSum = 0;
      let seen = 0;

      for (let b = 0; b < batches && !abort.aborted; b++) {
        const bt0 = now();
        const idx = order.slice(b * opts.batchSize, (b + 1) * opts.batchSize);
        const traceNow = opts.traceEvery > 0 && step % opts.traceEvery === 0 && !!hooks.onMechanics;

        const out = tf.tidy(() => {
          const idxT = tf.tensor1d(idx, 'int32');
          const xb0 = tf.gather(data.xTrain, idxT);
          const xb = data.batchAugment ? data.batchAugment(xb0, step) : xb0;
          const yb = tf.gather(yOneHot, idxT);
          const wb = tf.gather(sampleW, idxT);
          const before = traceNow ? vars.map((v) => v.clone()) : null;
          let accRef: tf.Tensor | null = null;

          // Forward pass + loss, recorded on the gradient tape.
          const { value, grads } = tf.variableGrads(() => {
            const logits = model.trainModel.apply(xb, { training: true }) as tf.Tensor;
            // variableGrads evaluates this closure in its own scope, so anything needed afterwards must be kept.
            accRef = tf.keep(tf.mean(tf.cast(tf.equal(tf.argMax(logits, 1), tf.argMax(yb, 1)), 'float32')));
            return weightedSoftmaxCE(logits, yb, wb);
          }, vars);

          const acc = accRef as unknown as tf.Tensor;
          const gradNorms = traceNow ? vars.map((v) => tf.norm(grads[v.name] as tf.Tensor)) : null;
          optimizer.applyGradients(grads as unknown as Record<string, tf.Variable>); // weight update
          const trace = traceNow && before && gradNorms
            ? vars.map((v, i) => tf.stack([gradNorms[i] as tf.Tensor, tf.norm(v), tf.norm(tf.sub(v, before[i] as tf.Tensor))]))
            : null;
          return { value, acc, trace: trace ? tf.stack(trace) : tf.zeros([1]) };
        });

        const [lossV, accV] = await Promise.all([out.value.data(), out.acc.data()]);
        const traceData = traceNow ? await out.trace.data() : null;
        out.value.dispose();
        out.acc.dispose();
        out.trace.dispose();

        const loss = lossV[0] as number;
        const accuracy = accV[0] as number;
        lossSum += loss * idx.length;
        accSum += accuracy * idx.length;
        seen += idx.length;
        const batchMs = now() - bt0;
        hooks.onBatchEnd?.({ epoch, batch: b + 1, batches, loss, accuracy, batchMs, samplesPerSec: (idx.length / batchMs) * 1000 });
        if (traceData) {
          const layers: LayerTrace[] = vars.map((v, i) => ({
            name: v.name,
            gradNorm: traceData[i * 3] as number,
            weightNorm: traceData[i * 3 + 1] as number,
            updateNorm: traceData[i * 3 + 2] as number,
          }));
          hooks.onMechanics?.({ step, epoch, loss, layers });
        }
        step++;
        if (b % 4 === 3) await yieldToEventLoop(); // let abort messages and UI events through
      }
      if (seen === 0) break;

      let valLoss: number | null = null;
      let valAccuracy: number | null = null;
      if (data.xVal && data.yVal && data.yVal.length > 0) {
        const v = await evaluateSplit(model.trainModel, data.xVal, data.yVal, C, opts.batchSize, data.valTransform);
        valLoss = v.loss;
        valAccuracy = v.accuracy;
        hooks.onValidationEnd?.({ epoch, valLoss, valAccuracy });
      }
      const epochMs = now() - et0;
      const m: EpochMetrics = { epoch, loss: lossSum / seen, accuracy: accSum / seen, valLoss, valAccuracy, epochMs, samplesPerSec: (seen / epochMs) * 1000 };
      history.push(m);
      hooks.onEpochEnd?.(m);

      const monitored = valLoss ?? m.loss;
      if (monitored < bestValLoss - 1e-4) {
        bestValLoss = monitored;
        bestEpoch = epoch;
        sinceBest = 0;
        disposeBest();
        bestWeights = model.trainModel.getWeights().map((w) => w.clone());
      } else {
        sinceBest++;
        if (opts.earlyStoppingPatience > 0 && sinceBest >= opts.earlyStoppingPatience) {
          stoppedEarly = true;
          break;
        }
      }
    }
    // Restore the best epoch (by validation loss when available).
    if (bestWeights && bestEpoch !== history.length) model.trainModel.setWeights(bestWeights);

    const result: EngineResult = { history, bestEpoch, stoppedEarly, aborted: abort.aborted, durationMs: now() - t0, paramCount, steps: step };
    hooks.onTrainingComplete?.(result);
    return result;
  } catch (e) {
    const err = e instanceof Error ? e : new Error(String(e));
    hooks.onTrainingError?.(err);
    throw err;
  } finally {
    disposeBest();
    yOneHot.dispose();
    sampleW.dispose();
    optimizer.dispose();
  }
}
