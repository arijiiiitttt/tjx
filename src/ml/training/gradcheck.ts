import * as tf from '@tensorflow/tfjs';
import type { BackendName, GradientCheckReport } from '@/types/ml';
import { mulberry32 } from '@/utils/rng';

/**
 * Verifies reverse-mode autodiff on the *active backend* by comparing analytic gradients
 * (tf.variableGrads == backpropagation) with numerical central differences on a tiny 2-layer network:
 *   dL/dw ≈ (L(w+ε) − L(w−ε)) / 2ε
 */
export async function gradientCheck(seed = 7): Promise<GradientCheckReport> {
  const rng = mulberry32(seed);
  const rand = (n: number) => Float32Array.from({ length: n }, () => rng() * 2 - 1);
  const [B, I, H, C] = [6, 5, 4, 3];
  const x = tf.tensor2d(rand(B * I), [B, I]);
  const y = tf.tidy(() => tf.oneHot(tf.tensor1d(Array.from({ length: B }, (_, i) => i % C), 'int32'), C));
  const mkVar = (name: string, n: number, shape: [number, number]) => {
    const init = tf.tensor2d(rand(n).map((v) => v * 0.6), shape);
    const v = tf.variable(init, true, name);
    init.dispose(); // tf.variable() does not take ownership of its initial tensor
    return v;
  };
  const setVar = (v: tf.Variable, values: Float32Array) => {
    const t = tf.tensor(values, v.shape);
    v.assign(t);
    t.dispose();
  };
  const w1 = mkVar('w1', I * H, [I, H]);
  const w2 = mkVar('w2', H * C, [H, C]);
  const loss = () => tf.tidy(() => tf.mean(tf.neg(tf.sum(tf.mul(y, tf.logSoftmax(tf.matMul(tf.tanh(tf.matMul(x, w1)), w2))), 1))) as tf.Scalar);
  const eps = 1e-2;
  const tolerance = 3e-2;
  let maxRel = 0;
  let checked = 0;
  const samples: GradientCheckReport['samples'] = [];
  try {
    const { value, grads } = tf.variableGrads(loss, [w1, w2]);
    value.dispose();
    for (const v of [w1, w2]) {
      const analytic = (await (grads[v.name] as tf.Tensor).data()) as Float32Array;
      const base = (await v.data()) as Float32Array;
      for (let k = 0; k < base.length; k += 2) {
        const plus = Float32Array.from(base);
        plus[k] = (plus[k] as number) + eps;
        setVar(v, plus);
        const lpT = loss();
        const lp = (await lpT.data())[0] as number;
        lpT.dispose();
        const minus = Float32Array.from(base);
        minus[k] = (minus[k] as number) - eps;
        setVar(v, minus);
        const lmT = loss();
        const lm = (await lmT.data())[0] as number;
        lmT.dispose();
        setVar(v, base);
        const numeric = (lp - lm) / (2 * eps);
        const a = analytic[k] as number;
        const rel = Math.abs(a - numeric) / Math.max(Math.abs(a), Math.abs(numeric), 1e-3);
        maxRel = Math.max(maxRel, rel);
        checked++;
        if (samples.length < 8) samples.push({ variable: `${v.name}[${k}]`, analytic: a, numeric });
      }
    }
    Object.values(grads).forEach((g) => g.dispose());
  } finally {
    x.dispose();
    y.dispose();
    w1.dispose();
    w2.dispose();
  }
  return { backend: tf.getBackend() as BackendName, passed: maxRel < tolerance, maxRelError: maxRel, checked, tolerance, samples };
}
