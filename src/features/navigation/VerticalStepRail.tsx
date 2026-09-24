import { Check, Lock } from 'lucide-react';
import { useApp } from '@/app/store';
import { STEPS } from './steps';
import { useStepGate } from './gate';

/**
 * A vertical echo of the top StepNav, floated in the empty grey margin beside the page on
 * desktop widths (≥1400px). Decorative only — the top nav remains the primary way to move
 * between steps. Shown on every page so it also gives a quick overall progress glance from
 * the Dashboard. Does not add a sidebar; it only uses the existing left grey margin.
 */
export function VerticalStepRail() {
  const page = useApp((s) => s.page);
  const setPage = useApp((s) => s.setPage);
  const { done, unlocked } = useStepGate();

  return (
    <ol className="pointer-events-none fixed left-4 top-1/2 z-10 hidden -translate-y-1/2 flex-col items-center gap-0 min-[1400px]:flex">
      {STEPS.map((step, i) => {
        const isDone = done[step.id];
        const active = page === step.id;
        const reachable = unlocked[step.id];
        return (
          <li key={step.id} className="pointer-events-auto flex flex-col items-center">
            <button
              tabIndex={-1}
              onClick={() => reachable && setPage(step.id)}
              disabled={!reachable}
              title={reachable ? step.label : `Finish the step before this to unlock ${step.label}`}
              className={`group flex flex-col items-center gap-1.5 rounded-md px-2 py-1.5 ${reachable ? 'cursor-pointer' : 'cursor-not-allowed'}`}
            >
              <span
                className={`grid h-7 w-7 shrink-0 place-items-center rounded-full text-xs font-semibold transition-colors ${
                  active
                    ? 'bg-accent-strong text-white'
                    : isDone
                    ? 'bg-good text-white'
                    : reachable
                    ? 'border border-line bg-surface text-mute group-hover:border-accent/60'
                    : 'border border-line bg-surface text-faint opacity-60'
                }`}
              >
                {isDone ? <Check className="h-3.5 w-3.5" aria-hidden /> : !reachable ? <Lock className="h-3 w-3" aria-hidden /> : step.n}
              </span>
              <span className={`text-[10px] font-medium tracking-wide ${active ? 'text-ink' : isDone ? 'text-mute' : 'text-faint'}`}>{step.label}</span>
            </button>
            {i < STEPS.length - 1 && <span className={`h-6 w-px ${isDone ? 'bg-good' : 'bg-line'}`} aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}
