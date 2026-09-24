import { ModelValidationError } from '@/ml/export/validate';
import { StorageError } from '@/storage/types';
import { MlRuntimeError } from '@/workers/rpc';

export interface Explained {
  title: string;
  detail: string;
  hint: string;
}

/** Turns raw failures into: what happened, why (where known), what the user can do. */
export function explainError(e: unknown, context = ''): Explained {
  const msg = e instanceof Error ? e.message : String(e);
  const name = (e as { name?: string })?.name ?? '';
  if (e instanceof ModelValidationError) return { title: 'Model import failed', detail: e.message, hint: e.hint };
  if (e instanceof StorageError) return { title: 'Storage problem', detail: e.message, hint: e.hint };
  if (/out of memory|oom|allocation failed|WebGL.*(lost|context)|GPU.*(memory|lost)|Failed to allocate|texture size/i.test(msg)) {
    return {
      title: 'The GPU ran out of memory',
      detail: `${msg}`,
      hint: 'Lower the batch size, use a smaller backbone (×0.5), use a smaller input size in scratch mode, remove images, or pin the WASM/CPU backend in Settings.',
    };
  }
  if (/backend/i.test(msg) && /could not be started|initialis/i.test(msg)) {
    return { title: 'No ML backend could start', detail: msg, hint: 'Update your browser, enable hardware acceleration, or try a different browser.' };
  }
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return { title: 'Camera permission denied', detail: 'The browser blocked access to the camera.', hint: 'Allow camera access in the address-bar site settings and try again. The page must be served over HTTPS or localhost.' };
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return { title: 'No camera found', detail: 'No video input device is available.', hint: 'Connect a camera or upload images instead.' };
  if (name === 'NotReadableError' || name === 'AbortError') return { title: 'Camera is busy', detail: 'The camera could not be opened, most likely because another app or tab is using it.', hint: 'Close other apps using the camera and try again.' };
  if (name === 'OverconstrainedError') return { title: 'Camera constraints not supported', detail: msg, hint: 'Choose a different camera in the dropdown.' };
  if (e instanceof MlRuntimeError) return { title: context || 'ML runtime error', detail: msg, hint: e.hint ?? 'Reload the page and try again with a smaller dataset or model.' };
  if (/download|TF Hub|network|fetch/i.test(msg)) return { title: 'Could not download the backbone', detail: msg, hint: 'This one-time download needs internet access. Afterwards it is cached and works offline. From-scratch mode needs no download.' };
  return { title: context || 'Unexpected error', detail: msg, hint: 'Try again. If it keeps happening, reload the page; your projects are stored locally and will still be there.' };
}
