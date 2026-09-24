import type { PageId } from '@/app/store';

export interface StepDef {
  id: PageId;
  n: number;
  label: string;
}

/** The five-step guided process. Dashboard (overview) and Settings are reached separately, not as steps. */
export const STEPS: StepDef[] = [
  { id: 'dataset', n: 1, label: 'Dataset' },
  { id: 'training', n: 2, label: 'Training' },
  { id: 'evaluation', n: 3, label: 'Evaluation' },
  { id: 'inference', n: 4, label: 'Inference' },
  { id: 'models', n: 5, label: 'Models' },
];

export const stepIndex = (id: PageId): number => STEPS.findIndex((s) => s.id === id);
export const isStep = (id: PageId): boolean => stepIndex(id) >= 0;
