import type { MlEvents } from '@/types/worker';

export interface RpcRequest {
  type: 'request';
  id: number;
  method: string;
  args: unknown[];
}
export type RpcResponse =
  | { type: 'response'; id: number; ok: true; result: unknown }
  | { type: 'response'; id: number; ok: false; error: SerializedError };
export interface RpcEvent {
  type: 'event';
  name: keyof MlEvents;
  payload: unknown;
}
export type WorkerMessage = RpcResponse | RpcEvent | { type: 'ready' };

export interface SerializedError {
  name: string;
  message: string;
  hint?: string;
}

export function serializeError(e: unknown): SerializedError {
  if (e instanceof Error) return { name: e.name, message: e.message, hint: (e as { hint?: string }).hint };
  return { name: 'Error', message: String(e) };
}

export class MlRuntimeError extends Error {
  constructor(message: string, readonly hint?: string, name = 'MlRuntimeError') {
    super(message);
    this.name = name;
  }
}
