import { Camera, Circle, Square, Video } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useApp } from '@/app/store';
import { Banner, Button } from '@/components/ui';
import { useCamera } from '@/hooks/useCamera';
import { uid } from '@/utils/download';

const REASON_TEXT: Record<string, string> = { blank: 'blank frame', 'too-dark': 'too dark', 'too-bright': 'too bright', duplicate: 'duplicate', 'too-small': 'too small', 'label-conflict': 'label conflict', corrupted: 'unreadable' };

export function CapturePanel({ classId, className }: { classId: string; className: string }) {
  const cam = useCamera();
  const ingestBlob = useApp((s) => s.ingestBlob);
  const [fps, setFps] = useState(4);
  const [recording, setRecording] = useState(false);
  const [stats, setStats] = useState<{ ok: number; rejected: Record<string, number> }>({ ok: 0, rejected: {} });
  const burst = useRef({ id: uid(), seq: 0, busy: false });

  const grab = useCallback(async () => {
    if (burst.current.busy) return;
    burst.current.busy = true;
    try {
      const blob = await cam.captureBlob();
      if (!blob) return;
      const seq = burst.current.seq++;
      const o = await ingestBlob(classId, blob, `camera-${burst.current.id.slice(0, 6)}-${String(seq).padStart(3, '0')}.jpg`, 'camera', burst.current.id, seq);
      setStats((s) => (o.accepted ? { ...s, ok: s.ok + 1 } : { ...s, rejected: { ...s.rejected, [o.reason ?? 'rejected']: (s.rejected[o.reason ?? 'rejected'] ?? 0) + 1 } }));
    } finally {
      burst.current.busy = false;
    }
  }, [cam, classId, ingestBlob]);

  useEffect(() => {
    if (!recording) return;
    const t = setInterval(() => void grab(), 1000 / fps);
    return () => clearInterval(t);
  }, [recording, fps, grab]);
  useEffect(() => { if (!cam.active) setRecording(false); }, [cam.active]);

  const toggleRecord = () => {
    if (!recording) burst.current = { id: uid(), seq: 0, busy: false }; // a new burst = a new leakage group
    setRecording((r) => !r);
  };
  const rejected = Object.entries(stats.rejected);

  return (
    <div className="space-y-3">
      {cam.error && <Banner tone="error" title={cam.error.title}>{cam.error.detail} <span className="block text-faint">→ {cam.error.hint}</span></Banner>}
      <div className="relative aspect-video overflow-hidden rounded-md border border-line bg-black">
        <video ref={cam.videoRef} muted playsInline className="h-full w-full -scale-x-100 object-cover" />
        {!cam.active && <div className="absolute inset-0 grid place-items-center text-sm text-mute"><Video className="mb-1 h-6 w-6" aria-hidden />Camera is off</div>}
        {recording && <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/60 px-2 py-0.5 text-xs text-bad"><Circle className="h-2.5 w-2.5 animate-pulse fill-current" aria-hidden /> Recording burst → {className}</div>}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {!cam.active ? (
          <Button variant="primary" icon={<Camera className="h-4 w-4" />} onClick={() => void cam.start()} disabled={cam.starting}>{cam.starting ? 'Starting…' : 'Start camera'}</Button>
        ) : (
          <>
            <Button onClick={() => void grab()} icon={<Camera className="h-4 w-4" />} disabled={recording}>Capture one</Button>
            <Button variant={recording ? 'danger' : 'primary'} onClick={toggleRecord} icon={recording ? <Square className="h-3.5 w-3.5" /> : <Circle className="h-3.5 w-3.5" />}>{recording ? 'Stop burst' : 'Record burst'}</Button>
            <Button variant="ghost" onClick={cam.stop}>Stop camera</Button>
          </>
        )}
        <label className="ml-auto flex items-center gap-2 text-xs text-mute">Burst rate
          <select className="rounded border border-line bg-bg px-1.5 py-1 text-ink" value={fps} onChange={(e) => setFps(Number(e.target.value))}>{[1, 2, 4, 8].map((f) => <option key={f} value={f}>{f}/s</option>)}</select>
        </label>
        {cam.devices.length > 1 && (
          <select aria-label="Camera device" className="rounded border border-line bg-bg px-1.5 py-1 text-xs text-ink" value={cam.deviceId} onChange={(e) => { cam.setDeviceId(e.target.value); }}>
            <option value="">Default camera</option>
            {cam.devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>)}
          </select>
        )}
      </div>
      <p className="text-xs text-faint">Frames are stored untouched (JPEG). Each burst is one leakage group: the split keeps neighbouring frames apart, but vary your pose, distance and background between bursts for a trustworthy validation score.</p>
      {(stats.ok > 0 || rejected.length > 0) && (
        <p className="text-xs text-mute">This session: <b className="text-good">{stats.ok} added</b>{rejected.length > 0 && <> · rejected {rejected.map(([k, v]) => `${v} ${REASON_TEXT[k] ?? k}`).join(', ')}</>}</p>
      )}
    </div>
  );
}
