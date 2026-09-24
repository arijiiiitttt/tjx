import { useApp } from '@/app/store';
import { analyzeDataset } from '@/ml/data/preflight';
import { STEPS, type StepDef } from './steps';
import type { PageId } from '@/app/store';

/**
 * Whether each step's own work is finished, and whether each step can be opened at all.
 * A step is "done" once its job is complete; a step is "unlocked" once every step before
 * it is done. Going backwards is always allowed by the caller — this only gates going
 * forward past a step that hasn't been finished yet.
 */
export function useStepGate(): { done: Record<PageId, boolean>; unlocked: Record<PageId, boolean> } {
  const project = useApp((s) => s.project);
  const samples = useApp((s) => s.samples);
  const experiments = useApp((s) => s.experiments);
  const models = useApp((s) => s.models);

  const classes = project?.classes ?? [];
  const analysis = analyzeDataset(classes, samples, { mode: 'transfer', validationSplit: 0.2 });

  const datasetDone = samples.length > 0 && !analysis.blocking;
  const trainingDone = experiments.length > 0;
  const evaluationDone = trainingDone; // evaluation metrics are computed automatically once training finishes
  const inferenceDone = models.length > 0;

  const doneByStep: Partial<Record<PageId, boolean>> = { dataset: datasetDone, training: trainingDone, evaluation: evaluationDone, inference: inferenceDone, models: false };

  const unlocked: Partial<Record<PageId, boolean>> = {};
  const done: Partial<Record<PageId, boolean>> = {};
  let prevOk = true;
  for (const step of STEPS as StepDef[]) {
    unlocked[step.id] = prevOk;
    done[step.id] = !!doneByStep[step.id];
    prevOk = prevOk && done[step.id]!;
  }

  return { done: done as Record<PageId, boolean>, unlocked: unlocked as Record<PageId, boolean> };
}
