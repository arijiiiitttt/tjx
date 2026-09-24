import { AlertTriangle } from 'lucide-react';
import { AUGMENTATION_GUIDE } from '@/ml/augmentation/config';
import { Toggle } from '@/components/ui';
import type { AugmentationConfig, RuntimeReport } from '@/types/ml';

interface SliderRow { key: 'rotationDeg' | 'zoom' | 'translate' | 'brightness' | 'contrast'; label: string; max: number; step: number; unit: string; scale?: number }
const SLIDERS: SliderRow[] = [
  { key: 'rotationDeg', label: 'Rotation', max: 45, step: 1, unit: '°' },
  { key: 'zoom', label: 'Zoom', max: 0.4, step: 0.01, unit: '%', scale: 100 },
  { key: 'translate', label: 'Translation', max: 0.3, step: 0.01, unit: '%', scale: 100 },
  { key: 'brightness', label: 'Brightness', max: 0.4, step: 0.01, unit: '%', scale: 100 },
  { key: 'contrast', label: 'Contrast', max: 0.5, step: 0.01, unit: '%', scale: 100 },
];

export function AugmentationPanel({ value, onChange, disabled, runtime }: { value: AugmentationConfig; onChange: (v: AugmentationConfig) => void; disabled?: boolean; runtime: RuntimeReport | null }) {
  const set = (patch: Partial<AugmentationConfig>) => onChange({ ...value, ...patch });
  const geometricBlocked = runtime && !runtime.affineAugmentation;
  return (
    <div className="space-y-3">
      <Toggle label="Enable augmentation (training images only)" checked={value.enabled} onChange={(v) => set({ enabled: v })} disabled={disabled} />
      {value.enabled && (
        <>
          <div>
            <Toggle label="Horizontal flip" checked={value.hFlip} onChange={(v) => set({ hFlip: v })} disabled={disabled} />
            {value.hFlip && <Risk text={AUGMENTATION_GUIDE[0]!.riskFor} />}
          </div>
          {SLIDERS.map((s) => {
            const g = AUGMENTATION_GUIDE.find((x) => x.key === s.key)!;
            const v = value[s.key];
            const disabledRow = !!disabled || (!!geometricBlocked && ['rotationDeg', 'zoom', 'translate'].includes(s.key));
            return (
              <div key={s.key} className={disabledRow ? 'opacity-50' : ''}>
                <div className="flex justify-between text-xs text-mute"><span>{s.label}</span><span className="tabular-nums text-ink">{v === 0 ? 'off' : `±${(v * (s.scale ?? 1)).toFixed(s.scale ? 0 : 0)}${s.unit}`}</span></div>
                <input type="range" className="w-full" min={0} max={s.max} step={s.step} value={v} disabled={disabledRow} onChange={(e) => set({ [s.key]: Number(e.target.value) } as Partial<AugmentationConfig>)} aria-label={s.label} />
                {v > 0 && <Risk text={g.riskFor} />}
              </div>
            );
          })}
          {geometricBlocked && <p className="text-xs text-warn">The active backend cannot run the affine-transform kernel; rotation, zoom and translation are skipped.</p>}
          {runtime && runtime.affineAugmentation && !runtime.affineBatch && <p className="text-xs text-faint">This backend's batched transform is unreliable (detected by the runtime probe); geometric augmentation runs per image, which is slower but correct.</p>}
        </>
      )}
    </div>
  );
}

const Risk = ({ text }: { text: string }) => (
  <p className="mt-0.5 flex items-start gap-1 text-[11px] text-faint"><AlertTriangle className="mt-0.5 h-3 w-3 shrink-0 text-warn/70" aria-hidden />Avoid if classes depend on: {text}.</p>
);
