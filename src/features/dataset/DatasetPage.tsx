import { AlertTriangle, ImagePlus, Plus, Trash2, Upload } from 'lucide-react';
import { useMemo, useRef, useState } from 'react';
import { useApp } from '@/app/store';
import { PALETTE } from '@/app/palette';
import { Badge, Banner, Button, Empty, PageHeader, Panel, ProgressBar, Segmented, Stat, inputCls } from '@/components/ui';
import { Thumb, forgetThumbs } from '@/components/Thumb';
import { CapturePanel } from '@/features/camera/CapturePanel';
import { WizardFooter } from '@/features/navigation/WizardFooter';
import { analyzeDataset, classStats, datasetFingerprint } from '@/ml/data/preflight';
import type { IngestOutcome } from '@/types/domain';
import { fmtBytes } from '@/utils/download';

const REASON: Record<string, string> = {
  corrupted: 'Corrupted / undecodable', 'unsupported-type': 'Not an image', 'too-large': 'File too large', blank: 'Blank image', 'too-dark': 'Too dark', 'too-bright': 'Too bright',
  'too-small': 'Resolution too low', duplicate: 'Duplicate', 'label-conflict': 'Already labelled as another class',
};

export function DatasetPage() {
  const { project, samples, addClass, renameClass, deleteClass, ingestFiles, deleteSamples, ingest, host } = useApp();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [report, setReport] = useState<IngestOutcome[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [showAll, setShowAll] = useState(false);
  const [view, setView] = useState<'grid' | 'list'>('grid');
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const classes = project?.classes ?? [];
  const cls = classes.find((c) => c.id === selectedId) ?? classes[0];
  const stats = useMemo(() => classStats(classes, samples), [classes, samples]);
  const analysis = useMemo(() => analyzeDataset(classes, samples, { mode: 'transfer', validationSplit: 0.2 }), [classes, samples]);
  const own = useMemo(() => samples.filter((s) => s.classId === cls?.id), [samples, cls]);
  const shown = showAll ? own : own.slice(-240);

  if (!project) return null;

  const onFiles = async (files: FileList | File[] | null) => {
    if (!files || !cls) return;
    const list = [...files];
    if (!list.length) return;
    setReport(await ingestFiles(cls.id, list));
  };
  const rejected = report?.filter((r) => !r.accepted) ?? [];
  const accepted = report?.filter((r) => r.accepted) ?? [];
  const warned = accepted.filter((r) => r.warnings.length);

  return (
    <>
      <PageHeader title="Dataset" subtitle="Build a labelled image dataset. Originals are stored untouched; quality checks, hashing and thumbnails are derived alongside them." />
      <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Images" value={samples.length.toLocaleString()} sub={fmtBytes(samples.reduce((a, s) => a + s.size, 0))} />
        <Stat label="Classes" value={classes.length} sub={`${stats.filter((s) => s.count > 0).length} with samples`} />
        <Stat label="Imbalance" value={`${analysis.imbalanceRatio.toFixed(1)}×`} tone={analysis.imbalanceRatio >= 2 ? 'warn' : 'good'} sub="largest / smallest class" />
        <Stat label="Dataset version" value={samples.length ? datasetFingerprint(samples) : '—'} sub="content fingerprint" />
      </div>

      {analysis.issues.filter((i) => i.severity !== 'info').length > 0 && (
        <div className="mb-4 space-y-2">
          {analysis.issues.filter((i) => i.severity !== 'info').map((i, k) => (
            <Banner key={k} tone={i.severity === 'error' ? 'error' : 'warning'} title={i.code === 'imbalance' ? '⚠ Class imbalance detected' : i.severity === 'error' ? 'Not trainable yet' : 'Dataset warning'}>{i.message}</Banner>
          ))}
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[260px_1fr]">
        <Panel title="Classes" action={<Button small icon={<Plus className="h-3.5 w-3.5" />} onClick={async () => { const c = await addClass(); if (c) setSelectedId(c.id); }}>Add</Button>} pad={false}>
          <ul className="divide-y divide-line">
            {stats.map((s) => {
              const c = classes.find((x) => x.id === s.classId)!;
              const active = c.id === cls?.id;
              return (
                <li key={c.id}>
                  <button onClick={() => { setSelectedId(c.id); setPicked(new Set()); setShowAll(false); }} className={`flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm ${active ? 'bg-raised' : 'hover:bg-raised/50'}`} aria-current={active}>
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: c.color }} aria-hidden />
                    <span className="min-w-0 flex-1 truncate">{c.name}</span>
                    <span className="tabular-nums text-mute">{s.count}</span>
                    {s.lowQuality > 0 && <span title={`${s.lowQuality} with quality warnings`}><AlertTriangle className="h-3.5 w-3.5 text-warn" aria-label="quality warnings" /></span>}
                  </button>
                </li>
              );
            })}
          </ul>
        </Panel>

        {cls ? (
          <div className="space-y-4">
            <Panel
              title={<span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: cls.color }} aria-hidden />{cls.name}</span>}
              action={<Button small variant="danger" icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => { if (confirm(`Delete class "${cls.name}" and its ${own.length} images?`)) { forgetThumbs(own.map((s) => s.id)); void deleteClass(cls.id); setSelectedId(null); } }}>Delete class</Button>}
            >
              <div className="grid gap-4 xl:grid-cols-2">
                <div className="space-y-3">
                  <label className="block text-xs text-mute">Class name
                    <input className={`${inputCls} mt-1`} defaultValue={cls.name} key={cls.id} onBlur={(e) => void renameClass(cls.id, e.target.value)} />
                  </label>
                  <div
                    onDragOver={(e) => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
                    onDrop={(e) => { e.preventDefault(); setDrag(false); void onFiles(e.dataTransfer.files); }}
                    className={`grid place-items-center rounded-md border border-dashed px-4 py-8 text-center text-sm ${drag ? 'border-accent bg-accent/10' : 'border-line'}`}
                  >
                    <Upload className="mb-2 h-5 w-5 text-mute" aria-hidden />
                    <p className="text-mute">Drop images here</p>
                    <Button className="mt-3" variant="primary" icon={<ImagePlus className="h-4 w-4" />} onClick={() => fileRef.current?.click()} disabled={!!ingest}>Choose images</Button>
                    <input ref={fileRef} type="file" accept="image/*" multiple hidden onChange={(e) => { void onFiles(e.target.files); e.target.value = ''; }} />
                  </div>
                  {ingest && <ProgressBar value={ingest.done / Math.max(1, ingest.total)} label={`Validating & storing ${ingest.done}/${ingest.total}`} />}
                  {report && !ingest && (
                    <div className="rounded-md border border-line bg-bg/60 p-3 text-xs">
                      <div className="font-medium text-ink">Import report</div>
                      <p className="mt-1 text-mute"><b className="text-good">{accepted.length} added</b>{warned.length > 0 && <> ({warned.length} with warnings)</>}{rejected.length > 0 && <> · <b className="text-bad">{rejected.length} rejected</b></>}</p>
                      {accepted.length > 0 && (
                        <ul className="mt-2 max-h-32 space-y-0.5 overflow-y-auto">
                          {accepted.map((r, i) => <li key={i} className="truncate text-ink" title={r.name}><span className="text-good">✓</span> {r.name}{r.warnings.length > 0 && <span className="text-warn"> · ⚠ {r.warnings.join(', ')}</span>}</li>)}
                        </ul>
                      )}
                      {rejected.length > 0 && (
                        <ul className="mt-2 max-h-40 space-y-1 overflow-y-auto">
                          {rejected.map((r, i) => <li key={i} className="text-mute"><span className="text-bad">{REASON[r.reason ?? ''] ?? r.reason}</span> — <span className="text-ink">{r.name}</span>: {r.detail}</li>)}
                        </ul>
                      )}
                    </div>
                  )}
                </div>
                <CapturePanel classId={cls.id} className={cls.name} />
              </div>
            </Panel>

            <Panel
              title={`Samples · ${own.length}`}
              action={
                <div className="flex items-center gap-2">
                  {picked.size > 0 && <Button small variant="danger" icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => { forgetThumbs([...picked]); void deleteSamples([...picked]); setPicked(new Set()); }}>Delete {picked.size}</Button>}
                  <Segmented value={view} onChange={setView} options={[{ value: 'grid', label: 'Grid' }, { value: 'list', label: 'File list' }]} />
                </div>
              }
            >
              {own.length === 0 ? <Empty>No samples yet. Upload images or use the camera above.</Empty> : view === 'grid' ? (
                <>
                  <div className="grid-thumbs">
                    {shown.map((s) => {
                      const on = picked.has(s.id);
                      const tip = `${s.name} · ${s.source} · ${s.width}×${s.height} · ${fmtBytes(s.size)}${s.quality.warnings.length ? ` · ⚠ ${s.quality.warnings.join(', ')}` : ''}`;
                      return (
                        <button key={s.id} className="cv-auto relative rounded" title={tip} aria-pressed={on} aria-label={`Sample ${tip}`}
                          onClick={() => setPicked((p) => { const n = new Set(p); on ? n.delete(s.id) : n.add(s.id); return n; })}>
                          <Thumb id={s.id} size={72} className={`w-full ${on ? 'opacity-40' : ''}`} ring={on ? PALETTE.bad : s.quality.warnings.length ? PALETTE.warn : undefined} />
                          {s.source === 'camera' && <span className="absolute bottom-0.5 left-0.5"><Badge>cam</Badge></span>}
                        </button>
                      );
                    })}
                  </div>
                  {own.length > shown.length && <Button className="mt-3" variant="ghost" onClick={() => setShowAll(true)}>Show all {own.length}</Button>}
                  <p className="mt-3 text-xs text-faint">Click thumbnails to select them for deletion. Amber outline = quality warning (low resolution, blur or near-duplicate).</p>
                </>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead><tr className="border-b border-line text-left text-xs text-mute">
                      <th className="w-8 py-1.5"><span className="sr-only">Select</span></th>
                      <th className="py-1.5 font-normal">File name</th><th className="font-normal">Source</th><th className="font-normal">Dimensions</th><th className="font-normal">Size</th><th className="font-normal">Added</th><th className="pr-2 text-right font-normal">Notes</th>
                    </tr></thead>
                    <tbody>
                      {[...own].reverse().map((s) => {
                        const on = picked.has(s.id);
                        return (
                          <tr key={s.id} className={`border-b border-line/60 last:border-0 ${on ? 'bg-bad/5' : ''}`}>
                            <td className="py-1.5"><input type="checkbox" aria-label={`Select ${s.name}`} checked={on} onChange={() => setPicked((p) => { const n = new Set(p); on ? n.delete(s.id) : n.add(s.id); return n; })} /></td>
                            <td className="max-w-[220px] truncate py-1.5 text-ink" title={s.name}>{s.name}</td>
                            <td className="text-mute">{s.source === 'camera' ? <Badge>camera</Badge> : 'upload'}</td>
                            <td className="text-mute tabular-nums">{s.width}×{s.height}</td>
                            <td className="text-mute tabular-nums">{fmtBytes(s.size)}</td>
                            <td className="text-mute">{new Date(s.createdAt).toLocaleString()}</td>
                            <td className="pr-2 text-right">{s.quality.warnings.length > 0 && <Badge tone="warn">{s.quality.warnings.join(', ')}</Badge>}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <p className="mt-3 text-xs text-faint">Every previously added file is listed here by name, newest first. Tick a row to select it for deletion.</p>
                </div>
              )}
            </Panel>
          </div>
        ) : <Empty>Add a class to begin.</Empty>}
      </div>
      {!host && <p className="mt-4 text-xs text-faint">The ML runtime is still starting; dataset editing works meanwhile.</p>}
      <WizardFooter page="dataset" note={samples.length === 0 ? 'Add some images before continuing to Training.' : undefined} />
    </>
  );
}
