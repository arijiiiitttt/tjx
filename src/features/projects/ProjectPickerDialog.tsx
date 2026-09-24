import { FolderOpen, Plus, Trash2 } from 'lucide-react';
import { useState } from 'react';
import { useApp } from '@/app/store';
import { Button, Modal, inputCls } from '@/components/ui';

/**
 * Shown once per app load, on top of everything else. It has no × button, no backdrop-click
 * dismiss and no Escape handler — the only way it closes is by picking "Continue" with a
 * project or creating a new one.
 */
export function ProjectPickerDialog() {
  const open = useApp((s) => s.projectPickerOpen);
  const projects = useApp((s) => s.projects);
  const project = useApp((s) => s.project);
  const openProject = useApp((s) => s.openProject);
  const createProject = useApp((s) => s.createProject);
  const deleteProject = useApp((s) => s.deleteProject);
  const closeProjectPicker = useApp((s) => s.closeProjectPicker);

  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [busy, setBusy] = useState(false);

  if (!open) return null;

  const hasExisting = projects.length > 0;
  const sel = selected ?? project?.id ?? projects[0]?.id ?? null;
  const selectedProject = projects.find((p) => p.id === sel);

  const continueWithSelected = async () => {
    if (!sel) return;
    setBusy(true);
    try {
      if (sel !== project?.id) await openProject(sel);
      closeProjectPicker();
    } finally {
      setBusy(false);
    }
  };

  const submitNew = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      await createProject(name);
      closeProjectPicker();
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (id: string, projectName: string) => {
    if (!confirm(`Delete project "${projectName}" and all its images, models and experiments? This cannot be undone.`)) return;
    setBusy(true);
    try {
      await deleteProject(id);
      if (selected === id) setSelected(null);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal title={hasExisting ? 'Welcome back' : 'Welcome to ML Studio'}>
      <p className="mb-4 text-sm text-mute">Each project keeps its own images, classes, models and experiments. Choose how you&apos;d like to start.</p>

      {hasExisting && !creating && (
        <div className="mb-4 space-y-2">
          <div className="text-xs font-medium text-mute">Continue with an existing project</div>
          <div className="max-h-48 space-y-1 overflow-y-auto rounded-md border border-line p-1">
            {projects.map((p) => (
              <label key={p.id} className={`flex cursor-pointer items-center gap-2 rounded-md px-2.5 py-2 text-sm ${sel === p.id ? 'bg-raised' : 'hover:bg-raised/50'}`}>
                <input type="radio" name="project-pick" className="accent-accent" checked={sel === p.id} onChange={() => setSelected(p.id)} disabled={busy} />
                <span className="min-w-0 flex-1 truncate">{p.name}</span>
                <span className="shrink-0 text-[11px] text-faint">{p.classes.length} classes</span>
                <button
                  type="button"
                  title={`Delete ${p.name}`}
                  aria-label={`Delete project ${p.name}`}
                  disabled={busy}
                  className="shrink-0 rounded p-1 text-faint transition-colors hover:bg-surface hover:text-bad disabled:opacity-40"
                  onClick={(e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    void handleDelete(p.id, p.name);
                  }}
                >
                  <Trash2 className="h-3.5 w-3.5" aria-hidden />
                </button>
              </label>
            ))}
          </div>
          <Button variant="primary" className="w-full justify-center" icon={<FolderOpen className="h-4 w-4" />} disabled={!sel || busy} onClick={() => void continueWithSelected()}>
            Continue with &quot;{selectedProject?.name}&quot;
          </Button>
        </div>
      )}

      {!creating ? (
        <Button variant={hasExisting ? 'secondary' : 'primary'} className="w-full justify-center" icon={<Plus className="h-4 w-4" />} onClick={() => { setCreating(true); setName(''); }}>
          Create a new project
        </Button>
      ) : (
        <div className={hasExisting ? 'space-y-2 border-t border-line pt-4' : 'space-y-2'}>
          <label className="block text-xs text-mute">
            Project name
            <input
              autoFocus
              className={`${inputCls} mt-1`}
              value={name}
              disabled={busy}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Cats vs dogs"
              onKeyDown={(e) => { if (e.key === 'Enter') void submitNew(); }}
            />
          </label>
          <div className="flex gap-2">
            {hasExisting && <Button variant="ghost" disabled={busy} onClick={() => setCreating(false)}>Back</Button>}
            <Button variant="primary" className="flex-1 justify-center" disabled={!name.trim() || busy} onClick={() => void submitNew()}>Create project</Button>
          </div>
        </div>
      )}
    </Modal>
  );
}