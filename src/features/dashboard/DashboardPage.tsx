import { ArrowRight, Check, Circle, Lock } from 'lucide-react';
import { useMemo } from 'react';
import { useApp } from '@/app/store';
import { Badge, Button, Empty, PageHeader, Panel, Stat } from '@/components/ui';
import { useStepGate } from '@/features/navigation/gate';
import { analyzeDataset, classStats } from '@/ml/data/preflight';
import { detectCapabilities } from '@/ml/runtime/capabilities';
import { fmtBytes, fmtMs, pct } from '@/utils/download';

export function DashboardPage() {
  const project = useApp((s) => s.project);
  const samples = useApp((s) => s.samples);
  const models = useApp((s) => s.models);
  const experiments = useApp((s) => s.experiments);
  const runtime = useApp((s) => s.runtime);
  const memory = useApp((s) => s.memory);
  const setPage = useApp((s) => s.setPage);
  const caps = useMemo(detectCapabilities, []);
  const { unlocked } = useStepGate();
  const classes = project?.classes ?? [];
  const analysis = useMemo(() => analyzeDataset(classes, samples, { mode: 'transfer', validationSplit: 0.2 }), [classes, samples]);
  const stats = classStats(classes, samples);
  const maxCount = Math.max(1, ...stats.map((s) => s.count));
  const best = experiments.reduce<typeof experiments[number] | null>((b, e) => ((e.finalMetrics.valAccuracy ?? -1) > (b?.finalMetrics.valAccuracy ?? -1) ? e : b), null);

  const steps = [
    { done: samples.length > 0, label: 'Add images to your classes', page: 'dataset' as const },
    { done: !analysis.blocking && samples.length > 0, label: 'Fix any dataset warnings', page: 'dataset' as const },
    { done: experiments.length > 0, label: 'Train a model', page: 'training' as const },
    { done: !!best, label: 'Evaluate it', page: 'evaluation' as const },
    { done: models.length > 0, label: 'Test it with an image or webcam', page: 'inference' as const },
  ].map((s) => ({ ...s, reachable: unlocked[s.page] }));
  const rt = (name: 'webgpu' | 'webgl' | 'wasm', browser: boolean) => {
    const a = runtime?.attempts.find((x) => x.backend === name);
    return a ? (a.ok ? <Badge tone="good">passed probe</Badge> : <span title={a.error}><Badge tone="bad">failed</Badge></span>) : <Badge tone={browser ? 'neutral' : 'bad'}>{browser ? 'available' : 'unavailable'}</Badge>;
  };

  return (
    <>
      <PageHeader title={project?.name ?? 'Dashboard'} subtitle="A local ML workstation: data, tensors, gradients and weights all live in this browser." />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Images" value={samples.length.toLocaleString()} sub={fmtBytes(samples.reduce((a, s) => a + s.size, 0))} />
        <Stat label="Experiments" value={experiments.length} sub={best ? `best val ${pct(best.finalMetrics.valAccuracy)}` : 'none yet'} />
        <Stat label="Saved models" value={models.length} />
        <Stat label="Active backend" value={runtime?.activeBackend ?? '…'} sub={runtime ? (runtime.host === 'worker' ? 'Web Worker' : 'main thread') : 'starting'} tone={runtime ? 'good' : undefined} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="Getting started" className="lg:col-span-1">
          <ol className="space-y-2">
            {steps.map((s, i) => (
              <li key={i}>
                <button
                  onClick={() => s.reachable && setPage(s.page)}
                  disabled={!s.reachable}
                  title={s.reachable ? undefined : 'Finish the previous step first'}
                  className={`flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-sm ${s.reachable ? 'hover:bg-raised' : 'cursor-not-allowed opacity-45'}`}
                >
                  {s.done ? <Check className="h-4 w-4 text-good" aria-label="done" /> : s.reachable ? <Circle className="h-4 w-4 text-faint" aria-label="todo" /> : <Lock className="h-4 w-4 text-faint" aria-label="locked" />}
                  <span className={s.done ? 'text-mute line-through' : ''}>{s.label}</span>
                  {s.reachable && <ArrowRight className="ml-auto h-3.5 w-3.5 text-faint" aria-hidden />}
                </button>
              </li>
            ))}
          </ol>
        </Panel>

        <Panel title="Dataset health" className="lg:col-span-1" action={<Button small variant="ghost" onClick={() => setPage('dataset')}>Open</Button>}>
          {samples.length === 0 ? <Empty>No images yet.</Empty> : (
            <ul className="space-y-2">
              {stats.map((s) => { const c = classes.find((x) => x.id === s.classId)!; return (
                <li key={s.classId}><div className="flex justify-between text-xs"><span>{s.name}</span><span className="tabular-nums text-mute">{s.count}</span></div>
                  <div className="mt-1 h-2 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full" style={{ width: `${(s.count / maxCount) * 100}%`, background: c.color }} /></div></li>); })}
            </ul>
          )}
          {analysis.issues.filter((i) => i.severity !== 'info').slice(0, 3).map((i, k) => <p key={k} className={`mt-2 text-xs ${i.severity === 'error' ? 'text-bad' : 'text-warn'}`}>{i.severity === 'error' ? '✕' : '⚠'} {i.message}</p>)}
        </Panel>

        <Panel title="Runtime" className="lg:col-span-1">
          <ul className="space-y-1.5 text-sm">
            <li className="flex justify-between"><span className="text-mute">WebGPU</span>{rt('webgpu', caps.webgpu)}</li>
            <li className="flex justify-between"><span className="text-mute">WebGL</span>{rt('webgl', caps.webgl2)}</li>
            <li className="flex justify-between"><span className="text-mute">WASM</span>{rt('wasm', caps.wasm)}</li>
          </ul>
          <dl className="mt-3 space-y-1 border-t border-line pt-3 text-xs">
            <div className="flex justify-between"><dt className="text-mute">Active backend</dt><dd className="font-medium">{runtime?.activeBackend ?? '—'}</dd></div>
            <div className="flex justify-between"><dt className="text-mute">Device</dt><dd className="max-w-[60%] truncate text-right">{runtime?.gpu ? [runtime.gpu.vendor, runtime.gpu.architecture, runtime.gpu.renderer].filter(Boolean).join(' ') : 'not exposed by the browser'}</dd></div>
            <div className="flex justify-between"><dt className="text-mute">Tensor memory</dt><dd className="tabular-nums">{memory ? `${fmtBytes(memory.numBytes)} · ${memory.numTensors} tensors` : '—'}</dd></div>
            {memory?.numBytesInGPU != null && <div className="flex justify-between"><dt className="text-mute">GPU memory</dt><dd className="tabular-nums">{fmtBytes(memory.numBytesInGPU)}</dd></div>}
          </dl>
        </Panel>

        <Panel title="Recent experiments" className="lg:col-span-3" pad={false}>
          {experiments.length === 0 ? <div className="p-4"><Empty>Trained runs will be listed here with their metrics.</Empty></div> : (
            <table className="w-full text-sm tabular-nums"><thead><tr className="border-b border-line text-left text-xs text-mute"><th className="px-4 py-2 font-normal">#</th><th className="font-normal">Architecture</th><th className="font-normal">Epochs</th><th className="font-normal">Val accuracy</th><th className="font-normal">Time</th><th className="pr-4 text-right font-normal">Backend</th></tr></thead>
              <tbody>{experiments.slice(0, 5).map((e) => <tr key={e.id} className="border-b border-line/60 last:border-0"><td className="px-4 py-2">#{String(e.number).padStart(3, '0')}</td><td>{e.architecture}</td><td>{e.epochsRun}</td><td>{pct(e.finalMetrics.valAccuracy)}</td><td>{fmtMs(e.durationMs)}</td><td className="pr-4 text-right text-mute">{e.backend}</td></tr>)}</tbody></table>
          )}
        </Panel>
      </div>
    </>
  );
}
