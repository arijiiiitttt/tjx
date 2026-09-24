import { AnimatePresence, motion } from 'framer-motion';
import { Activity, Cpu, LayoutDashboard, Lock, Pencil, Settings as SettingsIcon } from 'lucide-react';
import { useEffect } from 'react';
import { Banner } from '@/components/ui';
import { VerticalStepRail } from '@/features/navigation/VerticalStepRail';
import { DashboardPage } from '@/features/dashboard/DashboardPage';
import { DatasetPage } from '@/features/dataset/DatasetPage';
import { EvaluationPage } from '@/features/evaluation/EvaluationPage';
import { InferencePage } from '@/features/inference/InferencePage';
import { ModelsPage } from '@/features/models/ModelsPage';
import { ProjectPickerDialog } from '@/features/projects/ProjectPickerDialog';
import { SettingsPage } from '@/features/settings/SettingsPage';
import { TrainingPage } from '@/features/training/TrainingPage';
import { fmtBytes } from '@/utils/download';
import { useApp, type BootDeps, type PageId } from './store';

const PAGES: Record<PageId, () => React.JSX.Element | null> = {
  dashboard: DashboardPage, dataset: DatasetPage, training: TrainingPage, evaluation: EvaluationPage, inference: InferencePage, models: ModelsPage, settings: SettingsPage,
};

function StatusBar() {
  const { runtime, memory, host, project, projects, openProject, renameProject, training, refreshMemory } = useApp();
  useEffect(() => {
    if (!host) return;
    void refreshMemory();
    const t = setInterval(() => void refreshMemory(), 3000);
    return () => clearInterval(t);
  }, [host, refreshMemory]);
  const busy = training.status === 'preparing' || training.status === 'training' || training.status === 'finalizing';
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b border-line bg-raised/40 px-4 py-1.5 text-xs">
      <label className="flex items-center gap-1.5 text-mute">
        <span className="hidden sm:inline">Project</span>
        <select
          className="rounded-md border border-line bg-surface px-2 py-1 text-ink shadow-sm focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
          value={project?.id ?? ''}
          onChange={(e) => void openProject(e.target.value)}
          disabled={busy}
        >
          {projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
        {project && (
          <button
            title="Rename project"
            aria-label="Rename project"
            className="rounded-md p-1 text-faint transition-colors hover:bg-surface hover:text-ink"
            onClick={() => { const n = window.prompt('Project name', project.name); if (n && n.trim()) void renameProject(n); }}
          >
            <Pencil className="h-3 w-3" aria-hidden />
          </button>
        )}
      </label>

      <span className="hidden h-3.5 w-px bg-line sm:block" aria-hidden />

      <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-0.5 text-mute shadow-sm">
        <Cpu className="h-3 w-3 text-accent" aria-hidden />
        <span>
          <span className="text-faint">Backend</span>{' '}
          <b className="font-semibold text-ink">{runtime ? runtime.activeBackend : 'starting…'}</b>
          {runtime && <span className="text-faint"> · {runtime.host === 'worker' ? 'worker' : 'main'}</span>}
        </span>
      </span>

      {memory && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-surface px-2.5 py-0.5 text-mute shadow-sm tabular-nums">
          <span className="text-faint">Tensors</span>
          <b className="font-semibold text-ink">{memory.numTensors}</b>
          <span className="text-faint">·</span>
          <b className="font-semibold text-ink">{fmtBytes(memory.numBytes)}</b>
          {memory.numBytesInGPU != null && (
            <>
              <span className="text-faint">· GPU</span>
              <b className="font-semibold text-ink">{fmtBytes(memory.numBytesInGPU)}</b>
            </>
          )}
        </span>
      )}

      {busy && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-accent/30 bg-accent/10 px-2.5 py-0.5 font-medium text-accent">
          <Activity className="h-3 w-3 animate-pulse" aria-hidden />
          {training.status}…
        </span>
      )}

      <span className="ml-auto hidden items-center gap-1.5 text-faint md:flex">
        <Lock className="h-3 w-3 shrink-0" aria-hidden />
        <span>Images stay on device · training runs locally</span>
      </span>
    </div>
  );
}

function TopBar() {
  const page = useApp((s) => s.page);
  const setPage = useApp((s) => s.setPage);
  return (
    <header className="flex items-center justify-end gap-3 border-b border-line bg-surface px-4 py-2 shadow-[0_1px_0_0_rgba(20,22,28,0.04)]">
      <nav className="flex shrink-0 items-center gap-0.5 rounded-lg border border-line bg-raised/50 p-0.5" aria-label="App sections">
        <button
          onClick={() => setPage('dashboard')}
          aria-current={page === 'dashboard' ? 'page' : undefined}
          title="Dashboard"
          className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
            page === 'dashboard' ? 'bg-surface text-accent shadow-sm' : 'text-mute hover:bg-surface/80 hover:text-ink'
          }`}
        >
          <LayoutDashboard className="h-3.5 w-3.5" aria-hidden />
          <span className="hidden sm:inline">Dashboard</span>
        </button>
        <button
          onClick={() => setPage('settings')}
          aria-current={page === 'settings' ? 'page' : undefined}
          title="Settings"
          className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-xs font-medium transition-colors ${
            page === 'settings' ? 'bg-surface text-accent shadow-sm' : 'text-mute hover:bg-surface/80 hover:text-ink'
          }`}
        >
          <SettingsIcon className="h-3.5 w-3.5" aria-hidden />
          <span className="hidden sm:inline">Settings</span>
        </button>
      </nav>
    </header>
  );
}

export function App({ deps }: { deps?: BootDeps }) {
  const { ready, boot, page, notices, dismissNotice, storageWarning, projectPickerOpen } = useApp();
  useEffect(() => void boot(deps), [boot, deps]);
  const Page = PAGES[page];

  return (
    <div className="flex h-full flex-col">
      {ready && projectPickerOpen && <ProjectPickerDialog />}
      <VerticalStepRail />
      <TopBar />
      <StatusBar />
      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-[1280px] space-y-3 px-5 py-5">
          {storageWarning && <Banner tone="warning" title="Data will not be saved">{storageWarning}</Banner>}
          {notices.map((n) => (
            <Banner key={n.id} tone={n.kind} title={n.title} onClose={() => dismissNotice(n.id)}>
              {n.detail}{n.hint && <span className="mt-1 block text-faint">→ {n.hint}</span>}
            </Banner>
          ))}
          {!ready ? (
            <div className="py-24 text-center text-mute" role="status">Opening local storage…</div>
          ) : (
            <AnimatePresence mode="wait">
              <motion.div key={page} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.15 }}>
                <Page />
              </motion.div>
            </AnimatePresence>
          )}
        </div>
      </main>
    </div>
  );
}