import { ArrowLeft, ArrowRight, Flag } from 'lucide-react';
import { useApp } from '@/app/store';
import type { PageId } from '@/app/store';
import { Button } from '@/components/ui';
import { STEPS, stepIndex } from './steps';
import { useStepGate } from './gate';

/**
 * Back / Continue bar for the five-step guided process: finish this step, move to the next.
 * Placed at the bottom of each step page. The last step ("Models") shows "Finish" back to the
 * dashboard instead of "Continue".
 */
export function WizardFooter({ page, note }: { page: PageId; note?: string }) {
  const setPage = useApp((s) => s.setPage);
  const { unlocked } = useStepGate();
  const i = stepIndex(page);
  if (i < 0) return null;
  const prev = STEPS[i - 1];
  const next = STEPS[i + 1];
  const canContinue = !next || unlocked[next.id];

  return (
    <div className="mt-6 flex items-center justify-between gap-3 border-t border-line pt-4">
      <div>{prev ? <Button variant="ghost" icon={<ArrowLeft className="h-4 w-4" />} onClick={() => setPage(prev.id)}>Back to {prev.label}</Button> : <span />}</div>
      {note && <p className="hidden flex-1 text-center text-xs text-faint sm:block">{note}</p>}
      <div>
        {next ? (
          <Button
            variant="primary"
            disabled={!canContinue}
            title={canContinue ? undefined : `Finish this step before moving to ${next.label}`}
            onClick={() => canContinue && setPage(next.id)}
          >
            Continue to {next.label}<ArrowRight className="h-4 w-4" />
          </Button>
        ) : (
          <Button variant="primary" icon={<Flag className="h-4 w-4" />} onClick={() => setPage('dashboard')}>Finish · back to Dashboard</Button>
        )}
      </div>
    </div>
  );
}
