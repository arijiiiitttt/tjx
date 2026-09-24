import { HardDrive, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { useApp } from '@/app/store';
import { Badge, Button, Field, NumberInput, PageHeader, Panel, inputCls } from '@/components/ui';
import { detectCapabilities } from '@/ml/runtime/capabilities';
import type { StorageEstimate } from '@/storage/types';
import type { BackendPreference } from '@/types/ml';
import { fmtBytes, fmtMs } from '@/utils/download';

export function SettingsPage() {
  const settings = useApp((s) => s.settings);
  const update = useApp((s) => s.updateSettings);
  const runtime = useApp((s) => s.runtime);
  const storage = useApp((s) => s.storage);
  const perf = useApp((s) => s.perf);
  const clearAll = useApp((s) => s.clearAllData);
  const restart = useApp((s) => s.restartRuntime);
  const caps = useMemo(detectCapabilities, []);
  const [est, setEst] = useState<StorageEstimate | null>(null);
  useEffect(() => { void storage?.estimate().then(setEst); }, [storage]);

  return (
    <>
      <PageHeader title="Settings" subtitle="Runtime selection, storage and performance measurements." />
      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="ML backend" action={<Button small onClick={() => void restart()}>Restart runtime</Button>}>
          <Field label="Preferred backend" hint="'Auto' tries WebGPU → WebGL → WASM → CPU. A backend is only used if it passes a real gradient probe.">
            <select className={inputCls} value={settings.backend} onChange={(e) => void update({ backend: e.target.value as BackendPreference })}>
              <option value="auto">Auto (recommended)</option><option value="webgpu">WebGPU</option><option value="webgl">WebGL</option><option value="wasm">WebAssembly (CPU, SIMD)</option><option value="cpu">Plain JS CPU (slow)</option>
            </select>
          </Field>
          {runtime && (
            <div className="mt-4 text-sm">
              <div className="mb-1 text-xs text-mute">Runtime probe results ({runtime.host === 'worker' ? 'inside the Web Worker' : 'main thread'})</div>
              <ul className="space-y-1">
                {runtime.attempts.map((a) => (
                  <li key={a.backend} className="flex items-start gap-2 text-xs"><Badge tone={a.ok ? 'good' : 'bad'}>{a.backend}</Badge><span className="text-mute">{a.ok ? `passed in ${a.probeMs?.toFixed(0)} ms${runtime.activeBackend === a.backend ? ' — active' : ''}` : a.error}</span></li>
                ))}
              </ul>
              <p className="mt-2 text-xs text-faint">Affine augmentation kernel: {runtime.affineAugmentation ? (runtime.affineBatch ? 'available' : 'available per-image (batched kernel unreliable)') : 'unavailable'}.{runtime.gpu && ` GPU: ${[runtime.gpu.vendor, runtime.gpu.architecture, runtime.gpu.renderer].filter(Boolean).join(' ')}`}</p>
            </div>
          )}
        </Panel>

        <Panel title="Browser capabilities (reported by this browser)">
          <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
            {([['WebGPU', caps.webgpu], ['WebGL 2', caps.webgl2], ['WebAssembly', caps.wasm], ['WASM SIMD', caps.wasmSimd], ['WASM threads', caps.wasmThreads], ['Web Workers', caps.workers], ['OffscreenCanvas', caps.offscreenCanvas], ['IndexedDB', caps.indexedDB], ['createImageBitmap', caps.createImageBitmap], ['Camera API', caps.camera], ['Secure context', caps.secureContext]] as [string, boolean][]).map(([k, v]) => (
              <li key={k} className="flex items-center justify-between"><span className="text-mute">{k}</span><Badge tone={v ? 'good' : 'neutral'}>{v ? 'available' : 'unavailable'}</Badge></li>
            ))}
          </ul>
          <p className="mt-3 text-xs text-faint">WASM threads need cross-origin isolation (COOP/COEP headers); the app uses single-threaded WASM.</p>
        </Panel>

        <Panel title="Defaults">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Random seed"><NumberInput value={settings.seed} min={0} onChange={(v) => void update({ seed: Math.floor(v) })} /></Field>
            <Field label="Validation split"><NumberInput value={settings.validationSplit} min={0} max={0.5} step={0.05} onChange={(v) => void update({ validationSplit: v })} /></Field>
          </div>
          <p className="mt-2 text-xs text-faint">Applied to new training configurations. Switching mode on the Training page resets to these defaults.</p>
        </Panel>

        <Panel title="Storage">
          <div className="flex items-center gap-2 text-sm"><HardDrive className="h-4 w-4 text-mute" aria-hidden />{storage?.kind === 'indexeddb' ? 'IndexedDB' : <span className="text-warn">In-memory (not persistent)</span>}</div>
          {est && <p className="mt-2 text-sm text-mute">{fmtBytes(est.usage)} used of ~{fmtBytes(est.quota)} · {est.persisted ? 'persistent' : 'may be evicted under storage pressure'}</p>}
          <div className="mt-3 flex flex-wrap gap-2">
            {est && !est.persisted && <Button small onClick={async () => { await storage?.requestPersistence(); setEst(await storage?.estimate() ?? null); }}>Request persistent storage</Button>}
            <Button small variant="danger" icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => { if (confirm('Delete ALL projects, images, models and experiments from this browser?')) void clearAll(); }}>Delete all local data</Button>
          </div>
          <p className="mt-3 text-xs text-faint">After the app and the backbone have loaded once, training and inference work offline. The only network request the app can make is the one-time backbone download.</p>
        </Panel>

        <Panel title="Performance log (measured, this session)" className="lg:col-span-2" pad={false}>
          {perf.length === 0 ? <p className="p-4 text-sm text-mute">Measurements appear here after imports, training runs and model loads.</p> : (
            <table className="w-full text-sm tabular-nums"><tbody>{perf.map((p, i) => <tr key={i} className="border-b border-line/60 last:border-0"><td className="px-4 py-1.5 text-mute">{new Date(p.at).toLocaleTimeString()}</td><td>{p.label}</td><td className="pr-4 text-right">{fmtMs(p.ms)}</td></tr>)}</tbody></table>
          )}
        </Panel>
      </div>
    </>
  );
}
