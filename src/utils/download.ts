export function downloadBytes(name: string, bytes: Uint8Array | string, type = 'application/octet-stream'): void {
  const blob = new Blob([bytes as BlobPart], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export const uid = (): string => crypto.randomUUID();
export const fmtBytes = (n: number): string => (n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : n >= 1e6 ? `${(n / 1e6).toFixed(1)} MB` : n >= 1e3 ? `${(n / 1e3).toFixed(0)} KB` : `${n} B`);
export const fmtMs = (ms: number): string => (ms >= 60_000 ? `${(ms / 60_000).toFixed(1)} min` : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms.toFixed(0)} ms`);
export const pct = (v: number | null | undefined, d = 1): string => (v == null ? '—' : `${(v * 100).toFixed(d)}%`);
