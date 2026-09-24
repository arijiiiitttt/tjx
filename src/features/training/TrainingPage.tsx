import { Play, ShieldCheck, Square } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { defaultTrainingConfig } from '@/app/defaults';
import { PALETTE } from '@/app/palette';
import { useApp } from '@/app/store';
import { BarList, LineChart } from '@/components/charts';
import { Badge, Banner, Button, Field, NumberInput, PageHeader, Panel, ProgressBar, Segmented, Stat, Toggle, inputCls } from '@/components/ui';
import { analyzeDataset } from '@/ml/data/preflight';
import { BACKBONES } from '@/ml/models/backboneInfo';
import type { BackboneId, TrainingConfig, TrainingMode } from '@/types/ml';
import { fmtBytes, fmtMs, pct } from '@/utils/download';
import { AugmentationPanel } from './AugmentationPanel';
import { WizardFooter } from '@/features/navigation/WizardFooter';
import { CustomBackboneImport } from './CustomBackboneImport';

const PHASE_LABEL: Record<string, string> = {
  decoding: 'Decoding & preprocessing images', backbone: 'Loading backbone', features: 'Extracting embeddings (frozen backbone)', training: 'Training', evaluating: 'Evaluating on held-out split', finalizing: 'Serialising model',
};

export function TrainingPage() {
  const project = useApp((s) => s.project);
  const samples = useApp((s) => s.samples);
  const settings = useApp((s) => s.settings);
  const runtime = useApp((s) => s.runtime);
  const training = useApp((s) => s.training);
  const startTraining = useApp((s) => s.startTraining);
  const stopTraining = useApp((s) => s.stopTraining);
  const host = useApp((s) => s.host);
  const runGradCheck = useApp((s) => s.runGradientCheck);
  const gradCheck = useApp((s) => s.gradCheck);
  const customBackbones = useApp((s) => s.customBackbones);
  const [config, setConfig] = useState<TrainingConfig>(() => defaultTrainingConfig('transfer', settings));
  const [ack, setAck] = useState(false);
  const busy = ['preparing', 'training', 'finalizing'].includes(training.status);
  const set = (patch: Partial<TrainingConfig>) => setConfig((c) => ({ ...c, ...patch }));
  const analysis = useMemo(() => analyzeDataset(project?.classes ?? [], samples, config), [project, samples, config]);
  useEffect(() => setAck(false), [config.mode, samples.length]);

  const switchMode = (mode: TrainingMode) => setConfig({ ...defaultTrainingConfig(mode, settings), seed: config.seed, validationSplit: config.validationSplit });
  const canStart = !!host && !busy && !analysis.blocking && (!analysis.needsAcknowledgement || ack);
  const last = training.epochs[training.epochs.length - 1];
  const exp = useApp((s) => s.experiments.find((e) => e.id === training.lastExperimentId));
  const used = (project?.classes ?? []).filter((c) => samples.some((s) => s.classId === c.id));
  const arch = config.mode === 'transfer' ? BACKBONES[config.backboneId].label : `CNN ${config.scratchFilters.join('-')} · ${config.scratchInputSize}px`;

  return (
    <>
      <PageHeader title="Training" subtitle="Real forward passes, real gradients, real weight updates — running on your device. Nothing on this page is simulated." />
      <div className="grid gap-4 xl:grid-cols-[380px_1fr]">
        <div className="space-y-4">
          <Panel title="Model">
            <Segmented value={config.mode} onChange={switchMode} disabled={busy} options={[{ value: 'transfer', label: 'Transfer learning' }, { value: 'scratch', label: 'From scratch' }]} />
            {config.mode === 'transfer' ? (
              <div className="mt-3 space-y-3">
                <Field label="Pretrained backbone (frozen)" hint={config.customBackboneId ? 'A custom imported backbone is selected — needs no network.' : `${BACKBONES[config.backboneId].approxParams}. ${BACKBONES[config.backboneId].note} One-time download, then cached for offline use.`}>
                  <select className={inputCls} value={config.customBackboneId ?? config.backboneId} disabled={busy}
                    onChange={(e) => e.target.value.startsWith('custom:') ? set({ customBackboneId: e.target.value.slice(7) }) : set({ backboneId: e.target.value as BackboneId, customBackboneId: null })}>
                    <optgroup label="Built-in (TF Hub, downloaded once)">{Object.values(BACKBONES).map((b) => <option key={b.id} value={b.id}>{b.label}</option>)}</optgroup>
                    {customBackbones.length > 0 && <optgroup label="Imported (no network needed)">{customBackbones.map((b) => <option key={b.id} value={`custom:${b.id}`}>{b.name} ({b.featureDim}-d)</option>)}</optgroup>}
                  </select>
                </Field>
                <CustomBackboneImport disabled={busy} />
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Head hidden units"><NumberInput value={config.hiddenUnits} min={16} max={1024} step={16} disabled={busy} onChange={(v) => set({ hiddenUnits: v })} /></Field>
                  <Field label="Augmented copies / image" hint="Embeddings are cached, so augmentation = K frozen variants."><NumberInput value={config.augmentedCopies} min={0} max={10} disabled={busy} onChange={(v) => set({ augmentedCopies: v })} /></Field>
                </div>
              </div>
            ) : (
              <div className="mt-3 space-y-3">
                <Banner tone="info" title="Educational mode">A CNN trained from scratch needs far more data than transfer learning. Expect overfitting on small datasets.</Banner>
                <div className="grid grid-cols-2 gap-3">
                  <Field label="Input size (px)"><select className={inputCls} disabled={busy} value={config.scratchInputSize} onChange={(e) => set({ scratchInputSize: Number(e.target.value) })}>{[32, 48, 64, 96].map((n) => <option key={n}>{n}</option>)}</select></Field>
                  <Field label="Conv filters" hint="Per block, comma separated">
                    <input className={inputCls} disabled={busy} defaultValue={config.scratchFilters.join(',')} key={config.scratchFilters.join(',')}
                      onBlur={(e) => { const f = e.target.value.split(',').map((x) => parseInt(x, 10)).filter((x) => x > 0 && x <= 256).slice(0, 5); if (f.length) set({ scratchFilters: f }); }} />
                  </Field>
                  <Field label="Dense units"><NumberInput value={config.hiddenUnits} min={8} max={512} step={8} disabled={busy} onChange={(v) => set({ hiddenUnits: v })} /></Field>
                  <Field label="Mechanics trace every N steps"><NumberInput value={config.traceEvery} min={0} max={100} disabled={busy} onChange={(v) => set({ traceEvery: v })} /></Field>
                </div>
              </div>
            )}
          </Panel>

          <Panel title="Hyperparameters">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Epochs"><NumberInput value={config.epochs} min={1} max={500} disabled={busy} onChange={(v) => set({ epochs: v })} /></Field>
              <Field label="Batch size"><NumberInput value={config.batchSize} min={1} max={256} disabled={busy} onChange={(v) => set({ batchSize: v })} /></Field>
              <Field label="Learning rate"><NumberInput value={config.learningRate} min={0.00001} max={1} step={0.0005} disabled={busy} onChange={(v) => set({ learningRate: v })} /></Field>
              <Field label="Validation split"><NumberInput value={config.validationSplit} min={0} max={0.5} step={0.05} disabled={busy} onChange={(v) => set({ validationSplit: v })} /></Field>
              <Field label="Dropout"><NumberInput value={config.dropout} min={0} max={0.8} step={0.05} disabled={busy} onChange={(v) => set({ dropout: v })} /></Field>
              <Field label="Seed" hint="Split + augmentation"><NumberInput value={config.seed} min={0} step={1} disabled={busy} onChange={(v) => set({ seed: Math.floor(v) })} /></Field>
              <Field label="Early-stopping patience" hint="0 = off; best epoch is restored"><NumberInput value={config.earlyStoppingPatience} min={0} max={100} disabled={busy} onChange={(v) => set({ earlyStoppingPatience: v })} /></Field>
              <Field label="Class weighting">
                <select className={inputCls} value={config.classWeighting} disabled={busy} onChange={(e) => set({ classWeighting: e.target.value as 'none' | 'balanced' })}><option value="balanced">Balanced</option><option value="none">None</option></select>
              </Field>
            </div>
          </Panel>

          <Panel title="Augmentation"><AugmentationPanel value={config.augmentation} onChange={(a) => set({ augmentation: a })} disabled={busy} runtime={runtime} /></Panel>
        </div>

        <div className="min-w-0 space-y-4">
          <Panel title="Pre-flight dataset check" action={<Badge tone={analysis.blocking ? 'bad' : analysis.needsAcknowledgement ? 'warn' : 'good'}>{analysis.blocking ? 'blocked' : analysis.needsAcknowledgement ? 'warnings' : 'ready'}</Badge>}>
            {analysis.issues.length === 0 ? <p className="text-sm text-good">No problems found.</p> : (
              <ul className="space-y-1.5 text-[13px]">
                {analysis.issues.map((i, k) => <li key={k} className={i.severity === 'error' ? 'text-bad' : i.severity === 'warning' ? 'text-warn' : 'text-mute'}>{i.severity === 'error' ? '✕' : i.severity === 'warning' ? '⚠' : 'ℹ'} {i.message}</li>)}
              </ul>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-line pt-3">
              {analysis.needsAcknowledgement && !analysis.blocking && <Toggle label="I understand these warnings and want to train anyway" checked={ack} onChange={setAck} disabled={busy} />}
              <div className="ml-auto flex gap-2">
                {busy ? <Button variant="danger" icon={<Square className="h-3.5 w-3.5" />} onClick={stopTraining}>Stop & keep current weights</Button>
                  : <Button variant="primary" icon={<Play className="h-4 w-4" />} disabled={!canStart} onClick={() => void startTraining(config)}>Train model</Button>}
              </div>
            </div>
            {!host && <p className="mt-2 text-xs text-warn">The ML runtime is not running yet or failed to start (see notices above).</p>}
          </Panel>

          {(training.status !== 'idle') && (
            <Panel title="Live training" action={<Badge tone={training.status === 'error' ? 'bad' : training.status === 'done' ? 'good' : 'accent'}>{training.status}</Badge>}>
              {training.phase && training.phase !== 'training' && (
                <div className="mb-3"><ProgressBar value={training.backboneProgress ?? (training.phaseTotal ? training.phaseDone / training.phaseTotal : 0)} label={`${PHASE_LABEL[training.phase]}${training.phaseTotal ? ` (${training.phaseDone}/${training.phaseTotal})` : ''}`} /></div>
              )}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
                <Stat label="Model" value={arch} />
                <Stat label="Dataset" value={`${samples.filter((s) => used.some((c) => c.id === s.classId)).length.toLocaleString()} samples`} sub={`${used.length} classes`} />
                <Stat label="Epoch" value={`${training.currentEpoch || last?.epoch || 0} / ${training.totalEpochs || config.epochs}`} />
                <Stat label="Loss" value={last ? last.loss.toFixed(3) : '—'} />
                <Stat label="Accuracy" value={pct(last?.accuracy)} tone="good" />
                <Stat label="Val loss" value={last?.valLoss != null ? last.valLoss.toFixed(3) : '—'} />
                <Stat label="Val accuracy" value={pct(last?.valAccuracy)} tone="good" />
                <Stat label="Time / epoch" value={last ? fmtMs(last.epochMs) : '—'} sub={training.batch ? `batch ${fmtMs(training.batch.batchMs)}` : undefined} />
                <Stat label="Throughput" value={last ? `${last.samplesPerSec.toFixed(0)}/s` : '—'} sub="samples per second" />
                <Stat label="Backend" value={runtime?.activeBackend ?? '—'} sub={runtime?.host === 'worker' ? 'in Web Worker' : 'main thread'} />
              </div>
              {training.batch && training.status === 'training' && <div className="mt-3"><ProgressBar value={training.batch.batch / training.batch.batches} label={`Epoch ${training.batch.epoch} · batch ${training.batch.batch}/${training.batch.batches}`} /></div>}
              <div className="mt-4 grid gap-4 md:grid-cols-2">
                <LineChart title="Loss" series={[{ name: 'train', color: PALETTE.bad, values: training.epochs.map((e) => e.loss) }, { name: 'val', color: PALETTE.warn, dashed: true, values: training.epochs.map((e) => e.valLoss) }]} format={(v) => v.toFixed(2)} />
                <LineChart title="Accuracy" yMax={1} series={[{ name: 'train', color: PALETTE.sky, values: training.epochs.map((e) => e.accuracy) }, { name: 'val', color: PALETTE.good, dashed: true, values: training.epochs.map((e) => e.valAccuracy) }]} format={(v) => `${(v * 100).toFixed(0)}%`} />
              </div>
              {training.warnings.map((w, i) => <div key={i} className="mt-3"><Banner tone="warning" title="Note">{w}</Banner></div>)}
              {exp && training.status === 'done' && (
                <div className="mt-3 rounded-md border border-line bg-bg/60 p-3 text-sm">
                  <b>Experiment #{String(exp.number).padStart(3, '0')}</b> saved · {exp.epochsRun} epochs in {fmtMs(exp.durationMs)} · {exp.paramCount.toLocaleString()} trainable params · {fmtBytes(exp.modelBytes)}
                  {exp.stoppedEarly && ' · early-stopped'}{exp.aborted && ' · stopped by user'}. Validation accuracy <b>{pct(exp.finalMetrics.valAccuracy)}</b>, macro-F1 <b>{pct(exp.finalMetrics.macroF1)}</b>. See Evaluation for the confusion matrix.
                </div>
              )}
            </Panel>
          )}

          {config.mode === 'scratch' && (
            <Panel title="Training mechanics" action={<Button small icon={<ShieldCheck className="h-3.5 w-3.5" />} onClick={() => void runGradCheck()} disabled={!host || busy}>Verify backprop</Button>}>
              <ol className="mb-4 grid gap-2 text-xs text-mute sm:grid-cols-5">
                {[['1 Forward pass', 'images → conv/pool/dense → logits'], ['2 Loss', 'class-weighted cross-entropy vs. labels'], ['3 Gradient', 'tf.variableGrads records ops on a tape'], ['4 Backprop', 'tape replayed in reverse (chain rule)'], ['5 Update', 'Adam changes every weight']].map(([h, t]) => (
                  <li key={h} className="rounded-md border border-line bg-bg/60 p-2"><div className="font-medium text-ink">{h}</div>{t}</li>
                ))}
              </ol>
              <p className="mb-3 text-xs text-faint">Hand-writing convolution backprop in JavaScript would be ~100–1000× slower than the GPU kernels and would be a different, unverified implementation. TensorFlow.js runs the same reverse-mode autodiff on the GPU; "Verify backprop" proves it matches the mathematical definition.</p>
              {training.mechanics ? (
                <div>
                  <div className="mb-2 text-xs text-mute">Step {training.mechanics.step} · epoch {training.mechanics.epoch} · loss {training.mechanics.loss.toFixed(3)} — gradient L2 norm per parameter tensor</div>
                  <BarList max={Math.max(...training.mechanics.layers.map((l) => l.gradNorm), 1e-9)} format={(v) => v.toExponential(1)} rows={training.mechanics.layers.map((l) => ({ label: `${l.name.split('/').slice(-2).join('/')}  ·  |w|=${l.weightNorm.toFixed(2)}  Δ|w|=${l.updateNorm.toExponential(1)}`, value: l.gradNorm, color: PALETTE.accent }))} />
                </div>
              ) : <p className="text-xs text-faint">Start a from-scratch run to see live gradient and weight-update statistics.</p>}
              {gradCheck && (
                <div className={`mt-4 rounded-md border p-3 text-xs ${gradCheck.passed ? 'border-good/40 bg-good/10' : 'border-bad/40 bg-bad/10'}`}>
                  <b>{gradCheck.passed ? 'Gradient check passed' : 'Gradient check FAILED'}</b> on {gradCheck.backend}: max relative error {gradCheck.maxRelError.toExponential(2)} over {gradCheck.checked} weights (tolerance {gradCheck.tolerance}).
                  <table className="mt-2 w-full tabular-nums"><thead><tr className="text-left text-faint"><th>weight</th><th>autodiff</th><th>numerical</th></tr></thead>
                    <tbody>{gradCheck.samples.slice(0, 5).map((s) => <tr key={s.variable}><td>{s.variable}</td><td>{s.analytic.toFixed(5)}</td><td>{s.numeric.toFixed(5)}</td></tr>)}</tbody></table>
                </div>
              )}
            </Panel>
          )}
        </div>
      </div>
      <WizardFooter page="training" note={training.status !== 'done' ? 'Train a model, then continue to Evaluation.' : undefined} />
    </>
  );
}
