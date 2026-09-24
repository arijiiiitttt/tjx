import type { EvaluationReport } from '@/types/ml';
import { PALETTE } from '@/app/palette';

export function ConfusionMatrix({ report }: { report: EvaluationReport }) {
  const max = Math.max(1, ...report.confusion.flat());
  const names = report.classNames;
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1 text-xs" aria-label="Confusion matrix">
        <thead>
          <tr>
            <th className="p-1 text-right font-normal text-faint" rowSpan={2}>Actual ↓</th>
            <th className="pb-1 text-center font-normal text-faint" colSpan={names.length}>Predicted →</th>
          </tr>
          <tr>{names.map((n) => <th key={n} className="max-w-[88px] truncate px-1 text-center font-medium text-mute" title={n}>{n}</th>)}</tr>
        </thead>
        <tbody>
          {report.confusion.map((row, i) => (
            <tr key={names[i]}>
              <th className="max-w-[96px] truncate pr-2 text-right font-medium text-mute" title={names[i]}>{names[i]}</th>
              {row.map((v, j) => {
                const diag = i === j;
                const a = v / max;
                return (
                  <td key={j} className="h-11 w-16 rounded text-center font-semibold tabular-nums"
                    style={{ background: v === 0 ? PALETTE.raised : diag ? `rgba(21,127,74,${0.12 + a * 0.55})` : `rgba(201,42,62,${0.15 + a * 0.55})`, color: v === 0 ? PALETTE.faint : PALETTE.ink }}>
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
