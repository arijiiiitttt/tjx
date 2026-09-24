import { FileUp, Loader2, Trash2 } from 'lucide-react';
import { useRef, useState } from 'react';
import { useApp } from '@/app/store';
import { Button, Panel } from '@/components/ui';
import { fmtBytes } from '@/utils/download';

/**
 * Lets a person supply their own TF.js layers-format backbone (model.json + weights.bin) when the
 * built-in TF Hub download is unreachable (corporate proxy, air-gapped machine, restricted network).
 * The same layer/weight validation used for full model imports applies, then the file is actually
 * loaded once to confirm it produces a usable feature vector before being accepted.
 */
export function CustomBackboneImport({ disabled }: { disabled?: boolean }) {
  const customBackbones = useApp((s) => s.customBackbones);
  const importCustomBackbone = useApp((s) => s.importCustomBackbone);
  const deleteCustomBackbone = useApp((s) => s.deleteCustomBackbone);
  const ref = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const onFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setBusy(true);
    try {
      await importCustomBackbone([...files]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Panel title="Custom backbone (optional)" className="!bg-bg/40">
      <p className="text-xs text-faint">No network reachable? Select a TF.js layers-format <code>model.json</code> + <code>weights.bin</code> from anywhere (your own conversion, another machine, a mirror). It is validated the same way an imported model is, then loaded once to confirm it produces a usable feature vector.</p>
      <div className="mt-2 flex items-center gap-2">
        <Button small icon={busy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FileUp className="h-3.5 w-3.5" />} onClick={() => ref.current?.click()} disabled={disabled || busy}>{busy ? 'Validating…' : 'Import backbone files'}</Button>
        <input ref={ref} type="file" accept=".json,.bin" multiple hidden onChange={(e) => { void onFiles(e.target.files); e.target.value = ''; }} />
      </div>
      {customBackbones.length > 0 && (
        <ul className="mt-3 space-y-1.5">
          {customBackbones.map((b) => (
            <li key={b.id} className="flex items-center justify-between gap-2 rounded border border-line bg-surface px-2 py-1.5 text-xs">
              <span className="min-w-0 truncate">{b.name} <span className="text-faint">· {b.featureDim}-d · {b.inputSize}px · {fmtBytes(b.sizeBytes)}</span></span>
              <Button small variant="ghost" aria-label={`Delete backbone ${b.name}`} icon={<Trash2 className="h-3.5 w-3.5" />} onClick={() => { if (confirm(`Delete custom backbone "${b.name}"? Models trained with it will no longer load.`)) void deleteCustomBackbone(b.id); }} />
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
