import { Camera, ImageUp, Loader2 } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { useApp } from '@/app/store';
import { PALETTE } from '@/app/palette';
import { BarList } from '@/components/charts';
import { Badge, Banner, Button, Empty, PageHeader, Panel, Segmented, Stat, inputCls } from '@/components/ui';
import { useCamera } from '@/hooks/useCamera';
import { WizardFooter } from '@/features/navigation/WizardFooter';
import { decide } from '@/ml/inference/decision';
import type { PredictionResult } from '@/types/ml';
import { explainError } from '@/utils/errors';
import { pct } from '@/utils/download';

function Result({ result, threshold, colors }: { result: PredictionResult | null; threshold: number; colors: Map<string, string> }) {
  if (!result) return <p className="text-sm text-mute">No prediction yet.</p>;
  const d = decide(result.predictions, threshold);
  return (
    <div className="space-y-3">
      <div className={`rounded-md border px-3 py-2 ${d.isUnknown ? 'border-warn/40 bg-warn/10' : 'border-good/40 bg-good/10'}`} aria-live="polite">
        <div className="text-[11px] text-mute">Prediction (threshold {(threshold * 100).toFixed(0)}%)</div>
        <div className={`text-2xl font-semibold tracking-tight ${d.isUnknown ? 'text-warn' : 'text-good'}`}>{d.isUnknown ? 'UNKNOWN / LOW CONFIDENCE' : d.label?.toUpperCase()}</div>
        <div className="text-xs text-mute">top {pct(d.top.probability)} · margin {pct(d.margin, 0)} · entropy {d.entropy.toFixed(2)}</div>
      </div>
      <BarList rows={[...result.predictions].sort((a, b) => b.probability - a.probability).map((p, i) => ({ label: p.className, value: p.probability, strong: i === 0 && !d.isUnknown, color: i === 0 && !d.isUnknown ? colors.get(p.classId) ?? PALETTE.accent : undefined }))} format={(v) => `${(v * 100).toFixed(1)}%`} />
      <p className="text-xs text-mute">Latency <b className="tabular-nums text-ink">{result.latencyMs.toFixed(1)} ms</b> inference · <b className="tabular-nums text-ink">{result.preprocessMs.toFixed(1)} ms</b> preprocessing</p>
    </div>
  );
}

export function InferencePage() {
  const models = useApp((s) => s.models);
  const host = useApp((s) => s.host);
  const loadedModelId = useApp((s) => s.loadedModelId);
  const activeModelId = useApp((s) => s.activeModelId);
  const loadModel = useApp((s) => s.loadModel);
  const settings = useApp((s) => s.settings);
  const updateSettings = useApp((s) => s.updateSettings);
  const project = useApp((s) => s.project);
  const training = useApp((s) => s.training);
  const [tab, setTab] = useState<'image' | 'webcam'>('image');
  const [modelId, setModelId] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [live, setLive] = useState(false);
  const [effFps, setEffFps] = useState(0);
  const [error, setError] = useState<ReturnType<typeof explainError> | null>(null);
  const [loading, setLoading] = useState(false);
  const cam = useCamera();
  const busy = useRef(false);
  const stamps = useRef<number[]>([]);
  const id = modelId ?? activeModelId ?? models[0]?.id ?? null;
  const model = models.find((m) => m.id === id);
  const ready = !!id && loadedModelId === id;
  const colors = new Map(project?.classes.map((c) => [c.id, c.color]) ?? []);
  const training_ = ['preparing', 'training', 'finalizing'].includes(training.status);

  const ensureLoaded = async () => { if (!id || ready) return ready; setLoading(true); const ok = await loadModel(id); setLoading(false); return ok; };

  const predictOnce = async (src: ImageBitmapSource) => {
    if (!host) return;
    const bmp = await createImageBitmap(src, { imageOrientation: 'from-image' } as ImageBitmapOptions);
    return host.predictBitmap(bmp);
  };

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setError(null);
    try {
      if (!(await ensureLoaded())) return;
      setPreview((p) => { if (p) URL.revokeObjectURL(p); return URL.createObjectURL(f); });
      const r = await predictOnce(f);
      if (r) setResult(r);
    } catch (e) { setError(explainError(e, 'Could not classify this image')); }
  };

  // Camera stays smooth: frames are sampled at `inferenceFps`, transferred to the worker, and dropped while one is in flight.
  useEffect(() => {
    if (!live || !host) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const period = 1000 / settings.inferenceFps;
    const tick = async () => {
      if (stop) return;
      const t0 = performance.now();
      const v = cam.videoRef.current;
      if (v && v.readyState >= 2 && !busy.current) {
        busy.current = true;
        try {
          const r = await predictOnce(v);
          if (r && !stop) {
            setResult(r);
            const now = performance.now();
            stamps.current = [...stamps.current.filter((t) => now - t < 2000), now];
            setEffFps(stamps.current.length / 2);
          }
        } catch (e) { setError(explainError(e, 'Live inference stopped')); setLive(false); }
        finally { busy.current = false; }
      }
      timer = setTimeout(() => void tick(), Math.max(0, period - (performance.now() - t0)));
    };
    void tick();
    return () => { stop = true; clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, host, settings.inferenceFps]);
  useEffect(() => { if (!cam.active) setLive(false); }, [cam.active]);

  return (
    <>
      <PageHeader title="Inference" subtitle="Run a trained model locally on an image or the live camera. The exact preprocessing stored with the model is applied." />
      {models.length === 0 ? <Empty>No models yet. Train one or import one on the Models page.</Empty> : (
        <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
          <div className="space-y-4">
            <Panel>
              <div className="flex flex-wrap items-end gap-3">
                <label className="min-w-[240px] flex-1 text-xs text-mute">Model
                  <select className={`${inputCls} mt-1`} value={id ?? ''} onChange={(e) => { setModelId(e.target.value); setResult(null); setLive(false); }}>{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select>
                </label>
                <Segmented value={tab} onChange={(t) => { setTab(t); setLive(false); setResult(null); }} options={[{ value: 'image', label: 'Image' }, { value: 'webcam', label: 'Webcam' }]} />
                {ready ? <Badge tone="good">loaded</Badge> : <Button small onClick={() => void ensureLoaded()} disabled={loading || !host}>{loading ? <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading…</> : 'Load model'}</Button>}
              </div>
              {model && <p className="mt-2 text-xs text-faint">{model.metadata.modelType === 'scratch-cnn' ? 'CNN from scratch' : 'MobileNetV2 transfer'} · {model.metadata.inputShape[0]}px input · classes: {model.metadata.classes.map((c) => c.name).join(', ')}{model.metadata.backbone && ' · needs the cached backbone (downloaded once)'}</p>}
              {training_ && <p className="mt-2 text-xs text-warn">Training is running; inference will compete for the GPU.</p>}
            </Panel>
            {error && <Banner tone="error" title={error.title} onClose={() => setError(null)}>{error.detail}<span className="block text-faint">→ {error.hint}</span></Banner>}

            {tab === 'image' ? (
              <Panel title="Test an image">
                <label className="grid cursor-pointer place-items-center rounded-md border border-dashed border-line px-4 py-10 text-center text-sm text-mute hover:border-accent/60"
                  onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); void onFile(e.dataTransfer.files[0]); }}>
                  <ImageUp className="mb-2 h-6 w-6" aria-hidden />Drop an image or click to choose
                  <input type="file" accept="image/*" hidden onChange={(e) => { void onFile(e.target.files?.[0]); e.target.value = ''; }} />
                </label>
                {preview && <img src={preview} alt="Test input" className="mx-auto mt-4 max-h-80 rounded-md border border-line" />}
              </Panel>
            ) : (
              <Panel title="Live camera">
                {cam.error && <div className="mb-3"><Banner tone="error" title={cam.error.title}>{cam.error.detail}<span className="block text-faint">→ {cam.error.hint}</span></Banner></div>}
                <div className="relative aspect-video overflow-hidden rounded-md border border-line bg-black">
                  <video ref={cam.videoRef} muted playsInline className="h-full w-full -scale-x-100 object-cover" />
                  {!cam.active && <div className="absolute inset-0 grid place-items-center text-sm text-mute">Camera is off</div>}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  {!cam.active ? <Button variant="primary" icon={<Camera className="h-4 w-4" />} onClick={() => void cam.start()} disabled={cam.starting}>Start camera</Button> : <Button onClick={cam.stop}>Stop camera</Button>}
                  <Button variant={live ? 'danger' : 'primary'} disabled={!cam.active || !host} onClick={async () => { if (live) setLive(false); else if (await ensureLoaded()) setLive(true); }}>{live ? 'Stop inference' : 'Start inference'}</Button>
                  <label className="ml-auto flex items-center gap-2 text-xs text-mute">Inference rate
                    <input type="range" min={1} max={30} value={settings.inferenceFps} onChange={(e) => void updateSettings({ inferenceFps: Number(e.target.value) })} aria-label="Inference frames per second" />
                    <span className="w-14 tabular-nums text-ink">{settings.inferenceFps} FPS</span>
                  </label>
                </div>
                <p className="mt-2 text-xs text-faint">The camera renders at its native rate; frames are sampled at the chosen rate and skipped while the previous one is still being processed.</p>
              </Panel>
            )}
          </div>

          <div className="space-y-4">
            <Panel title="Result"><Result result={result} threshold={settings.threshold} colors={colors} /></Panel>
            {tab === 'webcam' && <div className="grid grid-cols-2 gap-3"><Stat label="Effective inference" value={live ? `${effFps.toFixed(1)} FPS` : '—'} /><Stat label="Latency" value={result ? `${result.latencyMs.toFixed(0)} ms` : '—'} /></div>}
            <Panel title="Confidence threshold">
              <input type="range" className="w-full" min={0.3} max={0.99} step={0.01} value={settings.threshold} onChange={(e) => void updateSettings({ threshold: Number(e.target.value) })} aria-label="Confidence threshold" />
              <div className="text-right text-sm tabular-nums">{(settings.threshold * 100).toFixed(0)}%</div>
              <p className="mt-2 text-xs text-faint">Softmax always distributes 100% over the known classes — even for an image unlike any training class — and is typically over-confident, so 90% does not mean "right 90% of the time". A threshold rejects hesitant outputs but cannot catch confident mistakes on unfamiliar inputs.</p>
            </Panel>
          </div>
        </div>
      )}
      <WizardFooter page="inference" />
    </>
  );
}
