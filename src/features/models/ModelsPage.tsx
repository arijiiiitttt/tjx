import { Download, FileUp, Package, Trash2 } from 'lucide-react';
import { useRef } from 'react';
import { useApp } from '@/app/store';
import { Badge, Button, Empty, PageHeader, Panel } from '@/components/ui';
import { fmtBytes, fmtMs, pct } from '@/utils/download';
import { WizardFooter } from '@/features/navigation/WizardFooter';

export function ModelsPage() {
  const models = useApp((s) => s.models);
  const experiments = useApp((s) => s.experiments);
  const loadedModelId = useApp((s) => s.loadedModelId);
  const exportModel = useApp((s) => s.exportModel);
  const importModel = useApp((s) => s.importModel);
  const deleteModel = useApp((s) => s.deleteModel);
  const deleteExperiment = useApp((s) => s.deleteExperiment);
  const loadModel = useApp((s) => s.loadModel);
  const setPage = useApp((s) => s.setPage);
  const fileRef = useRef<HTMLInputElement>(null);
  const bestVal = Math.max(-1, ...experiments.map((e) => e.finalMetrics.valAccuracy ?? -1));

  return (
    <>
      <PageHeader title="Models & experiments" subtitle="Export a model as TF.js files plus metadata, import one later, and compare every training run."
        actions={<><Button icon={<FileUp className="h-4 w-4" />} onClick={() => fileRef.current?.click()}>Import model</Button>
          <input ref={fileRef} type="file" multiple accept=".json,.bin,.zip" hidden onChange={(e) => { const f = [...(e.target.files ?? [])]; if (f.length) void importModel(f); e.target.value = ''; }} /></>} />
      <p className="-mt-3 mb-4 text-xs text-faint">Import either the exported <code>.zip</code> or <code>model.json</code> + <code>weights.bin</code> + <code>metadata.json</code> selected together. Imported files are treated as untrusted data: sizes, layer types and weight byte counts are validated, and nothing is executed.</p>

      <Panel title={`Models · ${models.length}`} pad={false} className="mb-4">
        {models.length === 0 ? <div className="p-4"><Empty>No models yet. <button className="text-accent underline" onClick={() => setPage('training')}>Train one</button> or import one.</Empty></div> : (
          <ul className="divide-y divide-line">
            {models.map((m) => (
              <li key={m.id} className="flex flex-wrap items-center gap-3 px-4 py-3">
                <Package className="h-4 w-4 text-accent" aria-hidden />
                <div className="min-w-[220px] flex-1">
                  <div className="flex flex-wrap items-center gap-2 text-sm font-medium">{m.name}{loadedModelId === m.id && <Badge tone="good">loaded</Badge>}{m.imported && <Badge tone="accent">imported</Badge>}</div>
                  <div className="text-xs text-mute">{m.metadata.classes.map((c) => c.name).join(' · ')} — {m.metadata.preprocessing.inputSize}px, {m.metadata.preprocessing.normalization.min}…{m.metadata.preprocessing.normalization.max}, {m.metadata.preprocessing.resize}</div>
                  <div className="text-xs text-faint">{new Date(m.createdAt).toLocaleString()} · {fmtBytes(m.sizeBytes)} · val acc {pct(m.metadata.finalMetrics.valAccuracy)} · macro-F1 {pct(m.metadata.finalMetrics.macroF1)}{m.metadata.backbone && ` · backbone ${m.metadata.backbone.id}`}</div>
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button small onClick={() => void loadModel(m.id)} disabled={loadedModelId === m.id}>Load</Button>
                  <Button small icon={<Download className="h-3.5 w-3.5" />} onClick={() => void exportModel(m.id, 'zip')}>.zip</Button>
                  <Button small variant="ghost" onClick={() => void exportModel(m.id, 'files')} title="Downloads model.json, weights.bin and metadata.json separately">3 files</Button>
                  <Button small variant="danger" aria-label={`Delete ${m.name}`} icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => { if (confirm(`Delete "${m.name}"?`)) void deleteModel(m.id); }} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </Panel>

      <Panel title={`Experiments · ${experiments.length}`} pad={false}>
        {experiments.length === 0 ? <div className="p-4"><Empty>Every training run is recorded here.</Empty></div> : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[980px] text-sm tabular-nums">
              <thead><tr className="border-b border-line text-left text-xs text-mute">
                {['#', 'Architecture', 'Dataset', 'Epochs', 'Batch', 'LR', 'Backend', 'Val acc', 'Macro F1', 'Time', 'Size', ''].map((h) => <th key={h} className="px-3 py-2 font-normal">{h}</th>)}</tr></thead>
              <tbody>
                {experiments.map((e) => {
                  const best = (e.finalMetrics.valAccuracy ?? -1) === bestVal && bestVal >= 0;
                  return (
                    <tr key={e.id} className="border-b border-line/60 last:border-0">
                      <td className="px-3 py-2 font-medium">#{String(e.number).padStart(3, '0')}</td>
                      <td className="px-3">{e.architecture}{e.aborted && <span className="ml-1 text-warn">(stopped)</span>}{e.stoppedEarly && <span className="ml-1 text-mute">(early stop)</span>}</td>
                      <td className="px-3 text-mute" title={`${e.classNames.join(', ')}`}>{e.datasetHash} · {e.sampleCount}</td>
                      <td className="px-3">{e.epochsRun}/{e.config.epochs}</td><td className="px-3">{e.config.batchSize}</td><td className="px-3">{e.config.learningRate}</td><td className="px-3">{e.backend}</td>
                      <td className={`px-3 font-medium ${best ? 'text-good' : ''}`}>{pct(e.finalMetrics.valAccuracy)}{best && ' ★'}</td><td className="px-3">{pct(e.finalMetrics.macroF1)}</td>
                      <td className="px-3">{fmtMs(e.durationMs)}</td><td className="px-3">{fmtBytes(e.modelBytes)}</td>
                      <td className="px-3 text-right"><Button small variant="ghost" aria-label={`Delete experiment ${e.number}`} icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => void deleteExperiment(e.id)} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      <WizardFooter page="models" />
    </>
  );
}
