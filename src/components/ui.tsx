import { AlertTriangle, Info, X, XCircle } from 'lucide-react';
import type { ButtonHTMLAttributes, ReactNode } from 'react';

const cx = (...a: (string | false | null | undefined)[]) => a.filter(Boolean).join(' ');

export function Panel({ title, action, children, className, pad = true }: { title?: ReactNode; action?: ReactNode; children: ReactNode; className?: string; pad?: boolean }) {
  return (
    <section className={cx('rounded-lg border border-line bg-surface', className)}>
      {(title || action) && (
        <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
          <h2 className="text-[13px] font-medium tracking-wide text-ink">{title}</h2>
          {action}
        </header>
      )}
      <div className={pad ? 'p-4' : ''}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: ReactNode; sub?: ReactNode; tone?: 'good' | 'warn' | 'bad' }) {
  const color = tone === 'good' ? 'text-good' : tone === 'warn' ? 'text-warn' : tone === 'bad' ? 'text-bad' : 'text-ink';
  return (
    <div className="min-w-0 rounded-md border border-line bg-bg/60 px-3 py-2">
      <div className="text-[11px] text-mute">{label}</div>
      <div className={cx('truncate text-lg font-semibold tabular-nums leading-tight', color)}>{value}</div>
      {sub && <div className="truncate text-[11px] text-faint">{sub}</div>}
    </div>
  );
}

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'ghost' | 'danger'; small?: boolean; icon?: ReactNode };
export function Button({ variant = 'secondary', small, icon, className, children, ...rest }: BtnProps) {
  const v = {
    primary: 'bg-accent-strong text-white hover:bg-accent-strong/85 border-transparent',
    secondary: 'bg-raised text-ink hover:border-accent/60 border-line',
    ghost: 'bg-transparent text-mute hover:text-ink border-transparent hover:bg-raised',
    danger: 'bg-transparent text-bad hover:bg-bad/10 border-bad/40',
  }[variant];
  return (
    <button {...rest} className={cx('inline-flex items-center justify-center gap-1.5 rounded-md border font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40', small ? 'px-2.5 py-1 text-xs' : 'px-3.5 py-1.5 text-sm', v, className)}>
      {icon}
      {children}
    </button>
  );
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'good' | 'warn' | 'bad' | 'accent' }) {
  const t = { neutral: 'border-line text-mute', good: 'border-good/40 text-good bg-good/10', warn: 'border-warn/40 text-warn bg-warn/10', bad: 'border-bad/40 text-bad bg-bad/10', accent: 'border-accent/40 text-accent bg-accent/10' }[tone];
  return <span className={cx('inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[11px] font-medium', t)}>{children}</span>;
}

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block text-xs text-mute">
      <span className="mb-1 block">{label}</span>
      {children}
      {hint && <span className="mt-1 block text-[11px] text-faint">{hint}</span>}
    </label>
  );
}

export const inputCls = 'w-full rounded-md border border-line bg-bg px-2.5 py-1.5 text-sm text-ink tabular-nums placeholder:text-faint disabled:opacity-50';

export function NumberInput({ value, onChange, min, max, step, disabled }: { value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; disabled?: boolean }) {
  return <input type="number" className={inputCls} value={value} min={min} max={max} step={step} disabled={disabled} onChange={(e) => Number.isFinite(e.target.valueAsNumber) && onChange(e.target.valueAsNumber)} />;
}

export function Toggle({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <label className={cx('flex cursor-pointer items-center gap-2 text-sm', disabled && 'opacity-50')}>
      <input type="checkbox" className="h-4 w-4 accent-accent" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      {label}
    </label>
  );
}

export function Segmented<T extends string>({ value, options, onChange, disabled }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; disabled?: boolean }) {
  return (
    <div role="radiogroup" className="inline-flex rounded-md border border-line bg-bg p-0.5">
      {options.map((o) => (
        <button key={o.value} role="radio" aria-checked={o.value === value} disabled={disabled} onClick={() => onChange(o.value)}
          className={cx('rounded px-3 py-1 text-sm transition-colors disabled:opacity-50', o.value === value ? 'bg-raised text-ink shadow-sm' : 'text-mute hover:text-ink')}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Banner({ tone, title, children, onClose }: { tone: 'error' | 'warning' | 'info'; title: string; children?: ReactNode; onClose?: () => void }) {
  const cfg = { error: ['border-bad/40 bg-bad/10', XCircle, 'text-bad'], warning: ['border-warn/40 bg-warn/10', AlertTriangle, 'text-warn'], info: ['border-sky/40 bg-sky/10', Info, 'text-sky'] }[tone] as [string, typeof Info, string];
  const Icon = cfg[1];
  return (
    <div role={tone === 'error' ? 'alert' : 'status'} className={cx('flex gap-3 rounded-md border px-3 py-2.5 text-sm', cfg[0])}>
      <Icon className={cx('mt-0.5 h-4 w-4 shrink-0', cfg[2])} aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="font-medium">{title}</div>
        {children && <div className="mt-0.5 text-[13px] text-mute">{children}</div>}
      </div>
      {onClose && (
        <button aria-label="Dismiss" onClick={onClose} className="text-mute hover:text-ink"><X className="h-4 w-4" /></button>
      )}
    </div>
  );
}

export function ProgressBar({ value, label }: { value: number; label?: string }) {
  return (
    <div>
      {label && <div className="mb-1 flex justify-between text-[11px] text-mute"><span>{label}</span><span className="tabular-nums">{Math.round(value * 100)}%</span></div>}
      <div className="h-1.5 overflow-hidden rounded-full bg-line"><div className="h-full rounded-full bg-accent transition-[width] duration-150" style={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }} /></div>
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: string; actions?: ReactNode }) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-0.5 max-w-2xl text-sm text-mute">{subtitle}</p>}
      </div>
      {actions}
    </div>
  );
}

export const Empty = ({ children }: { children: ReactNode }) => <div className="rounded-md border border-dashed border-line px-4 py-8 text-center text-sm text-mute">{children}</div>;

/**
 * A modal with no built-in way to dismiss it (no backdrop click, no Escape, no × button).
 * Use for choices the app requires an answer to before the person can continue.
 */
export function Modal({ title, children, className }: { title?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-black/60 p-4" role="presentation">
      <div role="dialog" aria-modal="true" aria-label={typeof title === 'string' ? title : undefined} className={cx('w-full max-w-md rounded-lg border border-line bg-surface p-5 shadow-xl', className)}>
        {title && <h2 className="mb-3 text-base font-semibold text-ink">{title}</h2>}
        {children}
      </div>
    </div>
  );
}
