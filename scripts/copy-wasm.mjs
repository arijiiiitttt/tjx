import { mkdirSync, copyFileSync, existsSync } from 'node:fs';
const src = 'node_modules/@tensorflow/tfjs-backend-wasm/dist/';
mkdirSync('public/wasm', { recursive: true });
for (const f of ['tfjs-backend-wasm.wasm', 'tfjs-backend-wasm-simd.wasm', 'tfjs-backend-wasm-threaded-simd.wasm']) {
  if (existsSync(src + f)) copyFileSync(src + f, 'public/wasm/' + f);
}
console.log('wasm binaries copied to public/wasm');
