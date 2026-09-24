import { useEffect, useState } from 'react';
import { useApp } from '@/app/store';

const cache = new Map<string, string>();

/** Loads a stored thumbnail into an object URL; revoked when the last user unmounts. */
export function useThumbUrl(sampleId: string): string | null {
  const storage = useApp((s) => s.storage);
  const [url, setUrl] = useState<string | null>(cache.get(sampleId) ?? null);
  useEffect(() => {
    let cancelled = false;
    if (cache.has(sampleId)) {
      setUrl(cache.get(sampleId) as string);
      return;
    }
    storage?.samples.getThumb(sampleId).then((b) => {
      if (!b || cancelled) return;
      const u = URL.createObjectURL(b);
      cache.set(sampleId, u);
      setUrl(u);
    });
    return () => {
      cancelled = true;
    };
  }, [sampleId, storage]);
  return url;
}

export function forgetThumbs(ids: string[]): void {
  ids.forEach((id) => {
    const u = cache.get(id);
    if (u) URL.revokeObjectURL(u);
    cache.delete(id);
  });
}

export function Thumb({ id, size = 72, className = '', title, ring }: { id: string; size?: number; className?: string; title?: string; ring?: string }) {
  const url = useThumbUrl(id);
  return url ? (
    <img src={url} alt="" title={title} width={size} height={size} loading="lazy" decoding="async" className={`aspect-square rounded object-cover ${className}`} style={ring ? { boxShadow: `0 0 0 2px ${ring}` } : undefined} />
  ) : (
    <div style={{ width: size, height: size }} className={`aspect-square animate-pulse rounded bg-raised ${className}`} />
  );
}
