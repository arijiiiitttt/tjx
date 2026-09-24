import type {
  BackboneId, BackendPreference, EvaluationReport, GradientCheckReport, MemoryReport, ModelArtifactsData, ModelMetadata, PredictionResult, RuntimeReport,
} from '@/types/ml';
import type { CustomBackboneInput, MlEvents, MlHost, RawSample, TrainRequest, TrainResponse } from '@/types/worker';
import { MlRuntimeError, type RpcRequest, type WorkerMessage } from './rpc';

type Listener = (payload: never) => void;

class Emitter {
  private listeners = new Map<string, Set<Listener>>();
  on<K extends keyof MlEvents>(name: K, cb: (p: MlEvents[K]) => void): () => void {
    const set = this.listeners.get(name) ?? new Set();
    set.add(cb as Listener);
    this.listeners.set(name, set);
    return () => set.delete(cb as Listener);
  }
  emit<K extends keyof MlEvents>(name: K, payload: MlEvents[K]): void {
    this.listeners.get(name)?.forEach((cb) => (cb as (p: MlEvents[K]) => void)(payload));
  }
}

/** Runs the ML runtime in a dedicated module Web Worker: the UI thread never blocks on tensors. */
export class WorkerHost extends Emitter implements MlHost {
  readonly kind = 'worker' as const;
  private worker: Worker;
  private nextId = 1;
  private pending = new Map<number, { resolve: (v: never) => void; reject: (e: Error) => void }>();

  constructor() {
    super();
    this.worker = new Worker(new URL('./ml.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<WorkerMessage>) => {
      const m = e.data;
      if (m.type === 'event') this.emit(m.name, m.payload as never);
      else if (m.type === 'response') {
        const p = this.pending.get(m.id);
        if (!p) return;
        this.pending.delete(m.id);
        if (m.ok) p.resolve(m.result as never);
        else p.reject(new MlRuntimeError(m.error.message, m.error.hint, m.error.name));
      }
    };
    const fail = (message: string) => {
      const err = new MlRuntimeError(message, 'The ML worker crashed (often out-of-memory). Reduce the dataset, batch size or model width and try again.');
      this.pending.forEach((p) => p.reject(err));
      this.pending.clear();
    };
    this.worker.onerror = (e) => fail(`The ML worker failed: ${e.message || 'unknown error'}`);
    this.worker.onmessageerror = () => fail('The ML worker sent an unreadable message.');
  }

  private call<T>(method: string, args: unknown[] = [], transfer: Transferable[] = []): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: never) => void, reject });
      const req: RpcRequest = { type: 'request', id, method, args };
      this.worker.postMessage(req, transfer);
    });
  }

  init = (pref: BackendPreference) => this.call<RuntimeReport>('init', [pref]);
  memory = () => this.call<MemoryReport>('memory');
  train = (req: TrainRequest) => this.call<TrainResponse>('train', [req]);
  loadModel = (a: ModelArtifactsData, m: ModelMetadata, cb?: CustomBackboneInput | null) => this.call<{ loadMs: number }>('loadModel', [a, m, cb ?? null]);
  unloadModel = () => this.call<void>('unloadModel');
  /** The bitmap is transferred (zero-copy) and closed by the worker. */
  predictBitmap = (b: ImageBitmap) => this.call<PredictionResult>('predictBitmap', [b], [b]);
  evaluate = (s: RawSample[]) => this.call<EvaluationReport>('evaluate', [s]);
  gradientCheck = () => this.call<GradientCheckReport>('gradientCheck');
  prefetchBackbone = (id: BackboneId) => this.call<{ cached: boolean; ms: number }>('prefetchBackbone', [id]);
  abortTraining(): void {
    this.worker.postMessage({ type: 'abort' });
  }
  terminate(): void {
    this.worker.terminate();
    this.pending.forEach((p) => p.reject(new MlRuntimeError('The ML runtime was stopped.')));
    this.pending.clear();
  }
}

/** Fallback: the identical MlService on the main thread (UI may stutter during training; the UI says so). */
export class InlineHost extends Emitter implements MlHost {
  readonly kind = 'inline' as const;
  private svc: Promise<import('@/ml/service').MlService>;

  constructor() {
    super();
    this.svc = import('@/ml/service').then((m) => new m.MlService((n, p) => this.emit(n, p), 'inline'));
  }
  private async s() {
    return this.svc;
  }
  init = async (pref: BackendPreference) => (await this.s()).init(pref);
  memory = async () => (await this.s()).memory();
  train = async (req: TrainRequest) => (await this.s()).train(req);
  loadModel = async (a: ModelArtifactsData, m: ModelMetadata, cb?: CustomBackboneInput | null) => (await this.s()).loadModel(a, m, cb ?? null);
  unloadModel = async () => (await this.s()).unloadModel();
  predictBitmap = async (b: ImageBitmap) => (await this.s()).predictBitmap(b);
  evaluate = async (s: RawSample[]) => (await this.s()).evaluate(s);
  gradientCheck = async () => (await this.s()).gradientCheck();
  prefetchBackbone = async (id: BackboneId) => (await this.s()).prefetchBackbone(id);
  abortTraining(): void {
    void this.s().then((s) => s.abortTraining());
  }
  terminate(): void {}
}

export const workerSupported = () => typeof Worker !== 'undefined' && typeof OffscreenCanvas !== 'undefined';

export interface HostInit {
  host: MlHost;
  report: RuntimeReport;
  /** Set when the worker could not be used and the runtime fell back to the main thread. */
  fallbackReason: string | null;
}

/** Starts the worker runtime; if the worker (not merely a backend) fails, retries inline. */
export async function startMlHost(pref: BackendPreference): Promise<HostInit> {
  let fallbackReason: string | null = null;
  if (workerSupported()) {
    const w = new WorkerHost();
    try {
      return { host: w, report: await w.init(pref), fallbackReason: null };
    } catch (e) {
      w.terminate();
      fallbackReason = `Worker runtime unavailable (${e instanceof Error ? e.message : String(e)}).`;
    }
  } else {
    fallbackReason = 'This browser lacks Web Workers with OffscreenCanvas, so the ML runtime runs on the main thread.';
  }
  const inline = new InlineHost();
  return { host: inline, report: await inline.init(pref), fallbackReason };
}
