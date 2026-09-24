import { Check, Lock } from 'lucide-react';
import { useApp } from '@/app/store';
import { STEPS, stepIndex } from './steps';
import { useStepGate } from './gate';

export function StepNav() {
  const page = useApp((s) => s.page);
  const setPage = useApp((s) => s.setPage);
  const current = stepIndex(page);
  const { unlocked } = useStepGate();

  return (
    <ol className="flex items-center" aria-label="Guided process">
      {STEPS.map((step, i) => {
        const done = i < current;
        const active = i === current;
        // Going back to an already-visited step is always fine; going forward needs the steps before it finished.
        const reachable = i <= current || unlocked[step.id];
        return (
          <li key={step.id} className="flex items-center">
            <button
              onClick={() => reachable && setPage(step.id)}
              disabled={!reachable}
              aria-current={active ? 'step' : undefined}
              aria-disabled={!reachable || undefined}
              className={`flex items-center gap-1.5 rounded-md px-1.5 py-1 text-sm transition-colors sm:px-2 ${
                reachable ? 'hover:bg-raised' : 'cursor-not-allowed opacity-45'
              }`}
              title={reachable ? step.label : `Finish the step before this to unlock ${step.label}`}
            >
              <span
                aria-hidden="true"
                className={`grid h-5 w-5 shrink-0 place-items-center rounded-full text-[11px] font-semibold ${
                  active ? 'bg-accent-strong text-white' : done ? 'bg-good text-white' : 'border border-line bg-surface text-mute'
                }`}
              >
                {done ? <Check className="h-3 w-3" aria-hidden /> : !reachable ? <Lock className="h-2.5 w-2.5" aria-hidden /> : step.n}
              </span>
              <span className={`hidden font-medium md:inline ${active ? 'text-ink' : done ? 'text-ink' : 'text-mute'}`}>{step.label}</span>
            </button>
            {i < STEPS.length - 1 && <span className={`mx-1 h-px w-4 sm:w-8 ${i < current ? 'bg-good' : 'bg-line'}`} aria-hidden />}
          </li>
        );
      })}
    </ol>
  );
}
