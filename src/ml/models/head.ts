import * as tf from '@tensorflow/tfjs';

/** A classifier expressed twice over the *same layers/weights*: logits (for a stable loss) and softmax (for export/inference). */
export interface ClassifierModel {
  trainModel: tf.LayersModel;
  exportModel: tf.LayersModel;
}

export interface HeadOptions {
  featureDim: number;
  numClasses: number;
  hiddenUnits: number;
  dropout: number;
}

export function buildHeadModel(o: HeadOptions): ClassifierModel {
  const input = tf.input({ shape: [o.featureDim], name: 'features' });
  let x = tf.layers.dropout({ rate: o.dropout }).apply(input) as tf.SymbolicTensor;
  x = tf.layers.dense({ units: o.hiddenUnits, activation: 'relu', kernelInitializer: 'heNormal' }).apply(x) as tf.SymbolicTensor;
  x = tf.layers.dropout({ rate: o.dropout }).apply(x) as tf.SymbolicTensor;
  const logits = tf.layers.dense({ units: o.numClasses, name: 'logits' }).apply(x) as tf.SymbolicTensor;
  const probs = tf.layers.softmax({ name: 'probabilities' }).apply(logits) as tf.SymbolicTensor;
  return {
    trainModel: tf.model({ inputs: input, outputs: logits }),
    exportModel: tf.model({ inputs: input, outputs: probs }),
  };
}
