import { PALETTE } from '@/app/palette';
import { RefreshCw } from 'lucide-react';
import { useMemo, useState } from 'react';
import { useApp } from '@/app/store';
import { Badge, Banner, Button, Empty, PageHeader, Panel, Segmented, Stat, inputCls } from '@/components/ui';
import { ConfusionMatrix } from '@/components/ConfusionMatrix';
import { Thumb } from '@/components/Thumb';
import { WizardFooter } from '@/features/navigation/WizardFooter';
import type { EvaluationReport } from '@/types/ml';
import type { RawSample } from '@/types/worker';
import { explainError } from '@/utils/errors';
import { pct } from '@/utils/download';

export function EvaluationPage() {
  const models = useApp((s) => s.models);
  const experiments = useApp((s) => s.experiments);
  const samples = useApp((s) => s.samples);
  const storage = useApp((s) => s.storage);
  const host = useApp((s) => s.host);
  const activeModelId = useApp((s) => s.activeModelId);
  const loadModel = useApp((s) => s.loadModel);
  const notify = useApp((s) => s.notify);
  const [modelId, setModelId] = useState<string | null>(null);
  const [scope, setScope] = useState<'validation' | 'all'>('validation');
  const [fresh, setFresh] = useState<{ modelId: string; scope: string; report: EvaluationReport } | null>(null);
  const [busy, setBusy] = useState(false);

  const id = modelId ?? activeModelId ?? models[0]?.id ?? null;
  const model = models.find((m) => m.id === id);
  const exp = experiments.find((e) => e.modelId === id);
  const report = fresh && fresh.modelId === id ? fresh.report : exp?.evaluation ?? null;
  const source = fresh && fresh.modelId === id ? `re-evaluated on ${fresh.scope}` : 'stored with the experiment (held-out validation split)';
  const classColor = useMemo(() => new Map(useApp.getState().project?.classes.map((c) => [c.id, c.color]) ?? []), [model]);
  const last = exp?.history[exp.history.length - 1];
  const gap = last && last.valAccuracy != null ? last.accuracy - last.valAccuracy : null;

  const run = async () => {
    if (!model || !host || !storage) return;
    setBusy(true);
    try {
      if (!(await loadModel(model.id))) return;
      const modelClasses = new Set(model.metadata.classes.map((c) => c.id));
      let pool = samples.filter((s) => modelClasses.has(s.classId));
      if (scope === 'validation' && exp) { const v = new Set(exp.valIds); pool = pool.filter((s) => v.has(s.id)); }
      if (!pool.length) { notify({ kind: 'warning', title: 'Nothing to evaluate', detail: 'No samples in this project match the model\'s classes (imported models can only be evaluated on projects with the same class ids).', hint: 'Use the Inference page to test imported models.' }); return; }
      const raw: RawSample[] = [];
      for (const s of pool) { const blob = await storage.samples.getBlob(s.id); if (blob) raw.push({ id: s.id, classId: s.classId, groupId: s.groupId, seq: s.seq, blob }); }
      const r = await host.evaluate(raw);
      setFresh({ modelId: model.id, scope: scope === 'validation' && exp ? 'the validation split' : 'all project samples (includes training data)', report: r });
    } catch (e) {
      const x = explainError(e, 'Evaluation failed');
      notify({ kind: 'error', title: x.title, detail: x.detail, hint: x.hint });
    } finally { setBusy(false); }
  };

  return (
    <>
      <PageHeader title="Evaluation" subtitle="Accuracy alone hides where a model fails. Inspect per-class precision/recall, the confusion matrix and the individual mistakes." />
      {models.length === 0 ? <Empty>Train a model first.</Empty> : (
        <>
          <Panel className="mb-4">
            <div className="flex flex-wrap items-end gap-3">
              <label className="min-w-[260px] flex-1 text-xs text-mute">Model
                <select className={`${inputCls} mt-1`} value={id ?? ''} onChange={(e) => setModelId(e.target.value)}>{models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select>
              </label>
              <div className="text-xs text-mute">Data<div className="mt-1"><Segmented value={scope} onChange={setScope} options={[{ value: 'validation', label: 'Validation split' }, { value: 'all', label: 'All samples' }]} /></div></div>
              <Button variant="primary" icon={<RefreshCw className={`h-4 w-4 ${busy ? 'animate-spin' : ''}`} />} onClick={() => void run()} disabled={busy || !host || !model}>{busy ? 'Evaluating…' : 'Run evaluation'}</Button>
            </div>
            {scope === 'all' && <p className="mt-2 text-xs text-warn">"All samples" includes training images, so the score is optimistic. Use it to find labelling mistakes, not to judge generalisation.</p>}
          </Panel>

          {!report ? <Empty>No evaluation is stored for this model. Press "Run evaluation".</Empty> : (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
                <Stat label="Accuracy" value={pct(report.accuracy)} tone={report.accuracy >= 0.9 ? 'good' : report.accuracy >= 0.7 ? 'warn' : 'bad'} />
                <Stat label="Macro F1" value={pct(report.macroF1)} sub="mean over classes" />
                <Stat label="Weighted F1" value={pct(report.weightedF1)} sub="weighted by support" />
                <Stat label="Samples" value={report.sampleCount} />
                <Stat label="Train − val accuracy" value={gap == null ? '—' : `${(gap * 100).toFixed(1)} pts`} tone={gap != null && gap > 0.15 ? 'bad' : 'good'} sub={gap != null && gap > 0.15 ? 'overfitting' : 'generalisation gap'} />
              </div>
              <p className="text-xs text-faint">Source: {source}.</p>
              {report.sampleCount < 30 && <Banner tone="warning" title="Very small evaluation set">With {report.sampleCount} samples, one mistake moves accuracy by {(100 / Math.max(1, report.sampleCount)).toFixed(1)} points. Treat these numbers as rough.</Banner>}
              <div className="grid gap-4 lg:grid-cols-2">
                <Panel title="Confusion matrix"><ConfusionMatrix report={report} /></Panel>
                <Panel title="Per-class performance" pad={false}>
                  <table className="w-full text-sm tabular-nums">
                    <thead><tr className="border-b border-line text-left text-xs text-mute"><th className="px-4 py-2 font-normal">Class</th><th className="font-normal">Precision</th><th className="font-normal">Recall</th><th className="font-normal">F1</th><th className="pr-4 text-right font-normal">Support</th></tr></thead>
                    <tbody>{report.perClass.map((c) => (
                      <tr key={c.classId} className="border-b border-line/60 last:border-0">
                        <td className="px-4 py-2"><span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: classColor.get(c.classId) ?? PALETTE.accent }} />{c.className}</td>
                        <td>{pct(c.precision, 0)}</td><td>{pct(c.recall, 0)}</td><td className={c.f1 < 0.7 ? 'text-bad' : ''}>{pct(c.f1, 0)}</td><td className="pr-4 text-right text-mute">{c.support}</td>
                      </tr>))}</tbody>
                  </table>
                </Panel>
              </div>
              <Panel title={`Most confident mistakes · ${report.misclassified.length}`}>
                {report.misclassified.length === 0 ? <p className="text-sm text-good">No misclassified samples in this set.</p> : (
                  <ul className="grid gap-3 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6">
                    {report.misclassified.map((m) => {
                      const cn = (cid: string) => report.classNames[report.classIds.indexOf(cid)] ?? '?';
                      return (
                        <li key={m.sampleId} className="rounded-md border border-line bg-bg/60 p-2">
                          <Thumb id={m.sampleId} size={112} className="w-full" ring={PALETTE.bad} />
                          <div className="mt-1.5 text-[11px] leading-tight"><span className="text-mute">true</span> {cn(m.trueClassId)}<br /><span className="text-bad">pred</span> {cn(m.predictedClassId)} <Badge tone="bad">{pct(m.probability, 0)}</Badge></div>
                        </li>
                      );
                    })}
                  </ul>
                )}
                <p className="mt-3 text-xs text-faint">A confident mistake is often a mislabelled image, a near-duplicate across classes, or a class that is genuinely ambiguous. Fix the dataset before tuning the model.</p>
              </Panel>
            </div>
          )}
        </>
      )}
      <WizardFooter page="evaluation" />
    </>
  );
}
