import * as tf from '@tensorflow/tfjs';
import type { BackendAttempt, BackendName, BackendPreference, RuntimeReport } from '@/types/ml';

export const BACKEND_ORDER: BackendName[] = ['webgpu', 'webgl', 'wasm', 'cpu'];

async function registerBackend(name: BackendName): Promise<void> {
  if (name === 'webgpu') {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) throw new Error('navigator.gpu is not available (WebGPU unsupported or disabled in this context).');
    await import('@tensorflow/tfjs-backend-webgpu');
  } else if (name === 'wasm') {
    if (typeof WebAssembly === 'undefined') throw new Error('WebAssembly is not available.');
    const wasm = await import('@tensorflow/tfjs-backend-wasm');
    const base = `${self.location.origin}${import.meta.env?.BASE_URL ?? '/'}wasm/`;
    wasm.setWasmPaths(base);
  }
  // 'webgl' and 'cpu' are registered by @tensorflow/tfjs itself.
}

export interface ProbeResult {
  affine: boolean;
  affineBatch: boolean;
}

/**
 * A real numerical check on the backend just selected: matmul + reverse-mode gradient with a known
 * answer, a conv2d forward/backward pass, and (optionally) the affine-transform kernel used for augmentation.
 * A backend that starts but computes wrong or throws is skipped.
 */
export async function probeBackend(): Promise<ProbeResult> {
  const A = tf.tensor2d([[1, 2], [3, 4]]);
  const w = tf.variable(tf.tensor2d([[1], [1]]));
  try {
    const { value, grads } = tf.variableGrads(() => tf.sum(tf.matMul(A, w)) as tf.Scalar, [w]);
    const v = (await value.data())[0] as number;
    const g = Array.from((await (grads[w.name] as tf.Tensor).data()) as Float32Array);
    if (Math.abs(v - 10) > 1e-2 || Math.abs((g[0] as number) - 4) > 1e-2 || Math.abs((g[1] as number) - 6) > 1e-2) {
      throw new Error(`gradient probe returned loss=${v}, grad=[${g.join(', ')}], expected 10 and [4, 6]`);
    }
    value.dispose();
    Object.values(grads).forEach((t) => t.dispose());

    const filt = tf.variable(tf.ones([3, 3, 1, 2]));
    const conv = tf.variableGrads(() => tf.sum(tf.conv2d(tf.ones([1, 6, 6, 1]) as tf.Tensor4D, filt as tf.Tensor4D, 1, 'same')) as tf.Scalar, [filt]);
    const cg = (await (conv.grads[filt.name] as tf.Tensor).data()) as Float32Array;
    if (!cg.every((x) => Number.isFinite(x)) || cg.length !== 18) throw new Error('conv2d gradient probe returned invalid values');
    conv.value.dispose();
    Object.values(conv.grads).forEach((t) => t.dispose());
    filt.dispose();

    // Affine transform kernel: single image, then a BATCH of two with different transforms.
    // (tfjs-backend-cpu 4.22 only transforms the first batch element; a batch-1 probe would not notice.)
    let affine = true;
    let affineBatch = true;
    try {
      const one = tf.ones([1, 8, 8, 3]) as tf.Tensor4D;
      const o1 = tf.image.transform(one, tf.tensor2d([[1, 0, 0, 0, 1, 0, 0, 0]]), 'bilinear', 'reflect', 0);
      const s1 = (await o1.data()) as Float32Array;
      affine = s1.length === 192 && s1.every((x) => Math.abs(x - 1) < 1e-3);
      tf.dispose([one, o1]);
      if (affine) {
        const ramp = tf.tensor4d(Float32Array.from({ length: 32 }, (_, i) => i % 4), [2, 4, 4, 1]);
        const ob = tf.image.transform(ramp, tf.tensor2d([[1, 0, 0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0, 0, 0]]), 'bilinear', 'reflect', 0);
        const d = (await ob.data()) as Float32Array;
        const first = Array.from(d.slice(0, 4));
        const second = Array.from(d.slice(16, 20));
        affineBatch = first.join() === '0,1,2,3' && second.join() === '1,2,3,3';
        tf.dispose([ramp, ob]);
      }
    } catch {
      affine = false;
      affineBatch = false;
    }
    return { affine, affineBatch };
  } finally {
    A.dispose();
    w.dispose();
  }
}

async function gpuInfo(): Promise<RuntimeReport['gpu']> {
  try {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      const gpu = (navigator as unknown as { gpu: { requestAdapter(): Promise<{ info?: { vendor?: string; architecture?: string } } | null> } }).gpu;
      const adapter = await gpu.requestAdapter();
      const info = adapter?.info;
      if (info && (info.vendor || info.architecture)) return { vendor: info.vendor || undefined, architecture: info.architecture || undefined };
    }
  } catch { /* fall through */ }
  try {
    if (typeof OffscreenCanvas !== 'undefined') {
      const gl = new OffscreenCanvas(1, 1).getContext('webgl') as WebGLRenderingContext | null;
      const ext = gl?.getExtension('WEBGL_debug_renderer_info');
      if (gl && ext) return { renderer: String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) };
    }
  } catch { /* not exposed */ }
  return null;
}

/** Tries backends in priority order (or the user's pinned one first) and returns what actually happened. */
export async function initBackend(pref: BackendPreference, host: 'worker' | 'inline', allowed: BackendName[] = BACKEND_ORDER): Promise<RuntimeReport> {
  const order = pref === 'auto' ? allowed : [pref, ...allowed.filter((b) => b !== pref)];
  const attempts: BackendAttempt[] = [];
  for (const name of order) {
    const t0 = performance.now();
    try {
      await registerBackend(name);
      const ok = await tf.setBackend(name);
      if (!ok) throw new Error('setBackend returned false (initialisation failed).');
      await tf.ready();
      const probe = await probeBackend();
      attempts.push({ backend: name, ok: true, probeMs: performance.now() - t0 });
      return { activeBackend: name, attempts, affineAugmentation: probe.affine, affineBatch: probe.affineBatch, host, gpu: name === 'cpu' || name === 'wasm' ? null : await gpuInfo() };
    } catch (e) {
      attempts.push({ backend: name, ok: false, error: e instanceof Error ? e.message : String(e), probeMs: performance.now() - t0 });
    }
  }
  throw new Error(`No TensorFlow.js backend could be started. Tried: ${attempts.map((a) => `${a.backend} (${a.error})`).join('; ')}`);
}

export function memoryReport() {
  const m = tf.memory() as tf.MemoryInfo & { numBytesInGPU?: number };
  return { numTensors: m.numTensors, numBytes: m.numBytes, numBytesInGPU: m.numBytesInGPU ?? null, unreliable: !!m.unreliable };
}
