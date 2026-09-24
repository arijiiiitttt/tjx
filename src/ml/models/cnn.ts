import * as tf from '@tensorflow/tfjs';
import type { ClassifierModel } from './head';

export interface CnnOptions {
  inputSize: number;
  filters: number[];
  numClasses: number;
  hiddenUnits: number;
  dropout: number;
}

/**
 * Small CNN: [Conv3x3 -> ReLU -> MaxPool2]xN -> Flatten -> Dropout -> Dense -> Dropout -> Logits (+ Softmax for export).
 * Batch-norm is deliberately omitted: with tiny datasets it adds moving-statistics state that complicates the
 * hand-written training loop without helping the educational goal.
 */
export function buildCnn(o: CnnOptions): ClassifierModel {
  const input = tf.input({ shape: [o.inputSize, o.inputSize, 3], name: 'image' });
  let x: tf.SymbolicTensor = input;
  o.filters.forEach((f) => {
    x = tf.layers.conv2d({ filters: f, kernelSize: 3, padding: 'same', activation: 'relu', kernelInitializer: 'heNormal' }).apply(x) as tf.SymbolicTensor;
    x = tf.layers.maxPooling2d({ poolSize: 2 }).apply(x) as tf.SymbolicTensor;
  });
  x = tf.layers.flatten().apply(x) as tf.SymbolicTensor;
  x = tf.layers.dropout({ rate: o.dropout }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.dense({ units: o.hiddenUnits, activation: 'relu', kernelInitializer: 'heNormal' }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.dropout({ rate: o.dropout }).apply(x) as tf.SymbolicTensor;
  const logits = tf.layers.dense({ units: o.numClasses, name: 'logits' }).apply(x) as tf.SymbolicTensor;
  const probs = tf.layers.softmax({ name: 'probabilities' }).apply(logits) as tf.SymbolicTensor;
  return {
    trainModel: tf.model({ inputs: input, outputs: logits }),
    exportModel: tf.model({ inputs: input, outputs: probs }),
  };
}
