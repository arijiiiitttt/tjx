import { PALETTE } from '@/app/palette';

export interface Series {
  name: string;
  color: string;
  values: (number | null)[];
  dashed?: boolean;
}

export function LineChart({ title, series, yMin = 0, yMax, format = (v: number) => v.toFixed(2), height = 170 }: { title: string; series: Series[]; yMin?: number; yMax?: number; format?: (v: number) => string; height?: number }) {
  const W = 420;
  const H = height;
  const pad = { l: 38, r: 8, t: 8, b: 20 };
  const n = Math.max(2, ...series.map((s) => s.values.length));
  const all = series.flatMap((s) => s.values).filter((v): v is number => v != null && Number.isFinite(v));
  const hi = yMax ?? (all.length ? Math.max(...all) * 1.08 || 1 : 1);
  const lo = yMin;
  const x = (i: number) => pad.l + ((W - pad.l - pad.r) * i) / (n - 1);
  const y = (v: number) => pad.t + (H - pad.t - pad.b) * (1 - (v - lo) / (hi - lo || 1));
  const ticks = [0, 0.5, 1].map((t) => lo + (hi - lo) * t);
  const last = series.map((s) => [...s.values].reverse().find((v) => v != null) ?? null);
  return (
    <figure className="min-w-0">
      <figcaption className="mb-1 flex flex-wrap items-center justify-between gap-2 text-xs text-mute">
        <span className="font-medium text-ink">{title}</span>
        <span className="flex gap-3">
          {series.map((s, i) => (
            <span key={s.name} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-3" style={{ background: s.color }} />
              {s.name} <span className="tabular-nums text-ink">{last[i] != null ? format(last[i] as number) : '—'}</span>
            </span>
          ))}
        </span>
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={`${title} chart`}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.l} x2={W - pad.r} y1={y(t)} y2={y(t)} stroke={PALETTE.line} strokeWidth="1" />
            <text x={pad.l - 6} y={y(t) + 3.5} textAnchor="end" fontSize="10" fill={PALETTE.faint}>{format(t)}</text>
          </g>
        ))}
        <text x={pad.l} y={H - 5} fontSize="10" fill={PALETTE.faint}>epoch 1</text>
        <text x={W - pad.r} y={H - 5} fontSize="10" fill={PALETTE.faint} textAnchor="end">{n}</text>
        {series.map((s) => {
          let d = '';
          s.values.forEach((v, i) => {
            if (v == null || !Number.isFinite(v)) return;
            d += `${d && s.values[i - 1] != null ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
          });
          return d ? <path key={s.name} d={d} fill="none" stroke={s.color} strokeWidth="2" strokeDasharray={s.dashed ? '5 4' : undefined} strokeLinejoin="round" /> : null;
        })}
      </svg>
    </figure>
  );
}

export function BarList({ rows, max = 1, format = (v: number) => `${(v * 100).toFixed(0)}%` }: { rows: { label: string; value: number; color?: string; strong?: boolean }[]; max?: number; format?: (v: number) => string }) {
  return (
    <ul className="space-y-1.5">
      {rows.map((r) => (
        <li key={r.label}>
          <div className={`flex justify-between text-xs ${r.strong ? 'font-semibold text-ink' : 'text-mute'}`}><span className="truncate">{r.label}</span><span className="tabular-nums">{format(r.value)}</span></div>
          <div className="h-2 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full" style={{ width: `${Math.min(100, (r.value / max) * 100)}%`, background: r.color ?? (r.strong ? PALETTE.accent : PALETTE.faint) }} /></div>
        </li>
      ))}
    </ul>
  );
}
