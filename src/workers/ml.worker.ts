/// <reference lib="webworker" />
import { MlService } from '@/ml/service';
import type { MlEvents, TrainResponse } from '@/types/worker';
import type { RpcRequest, RpcResponse, WorkerMessage } from './rpc';
import { serializeError } from './rpc';

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: WorkerMessage, transfer: Transferable[] = []) => ctx.postMessage(m, transfer);

const service = new MlService(
  <K extends keyof MlEvents>(name: K, payload: MlEvents[K]) => post({ type: 'event', name, payload }),
  'worker',
);

const methods: Record<string, (...args: never[]) => Promise<unknown>> = {
  init: service.init.bind(service),
  memory: service.memory.bind(service),
  train: service.train.bind(service),
  loadModel: service.loadModel.bind(service),
  unloadModel: service.unloadModel.bind(service),
  predictBitmap: service.predictBitmap.bind(service),
  evaluate: service.evaluate.bind(service),
  gradientCheck: service.gradientCheck.bind(service),
  prefetchBackbone: service.prefetchBackbone.bind(service),
};

ctx.onmessage = async (e: MessageEvent<RpcRequest | { type: 'abort' }>) => {
  const msg = e.data;
  if ('type' in msg && msg.type === 'abort') {
    service.abortTraining(); // handled between awaits inside the training loop
    return;
  }
  const req = msg as RpcRequest;
  try {
    const fn = methods[req.method];
    if (!fn) throw new Error(`Unknown method ${req.method}`);
    const result = await (fn as (...a: unknown[]) => Promise<unknown>)(...req.args);
    const transfer: Transferable[] = [];
    if (req.method === 'train') transfer.push((result as TrainResponse).artifacts.weightData);
    const res: RpcResponse = { type: 'response', id: req.id, ok: true, result };
    post(res, transfer);
  } catch (err) {
    post({ type: 'response', id: req.id, ok: false, error: serializeError(err) });
  }
};

post({ type: 'ready' });
