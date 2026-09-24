export interface BrowserCapabilities {
  webgpu: boolean;
  webgl2: boolean;
  wasm: boolean;
  wasmSimd: boolean;
  wasmThreads: boolean;
  workers: boolean;
  offscreenCanvas: boolean;
  indexedDB: boolean;
  createImageBitmap: boolean;
  camera: boolean;
  secureContext: boolean;
}

// Smallest valid module using a SIMD instruction (v128.const + drop) – used by the TF.js/Emscripten feature test.
const SIMD_TEST = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

/** Main-thread feature detection: only facts the browser actually reports. */
export function detectCapabilities(): BrowserCapabilities {
  const wasm = typeof WebAssembly === 'object';
  let webgl2 = false;
  try {
    webgl2 = !!document.createElement('canvas').getContext('webgl2');
  } catch { /* unavailable */ }
  let wasmSimd = false;
  try {
    wasmSimd = wasm && WebAssembly.validate(SIMD_TEST);
  } catch { /* unavailable */ }
  return {
    webgpu: typeof navigator !== 'undefined' && 'gpu' in navigator,
    webgl2,
    wasm,
    wasmSimd,
    wasmThreads: wasm && typeof SharedArrayBuffer !== 'undefined' && typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated,
    workers: typeof Worker !== 'undefined',
    offscreenCanvas: typeof OffscreenCanvas !== 'undefined',
    indexedDB: typeof indexedDB !== 'undefined',
    createImageBitmap: typeof createImageBitmap === 'function',
    camera: !!navigator.mediaDevices?.getUserMedia,
    secureContext: typeof isSecureContext !== 'undefined' && isSecureContext,
  };
}
