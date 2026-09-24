import { useCallback, useEffect, useRef, useState } from 'react';
import { explainError, type Explained } from '@/utils/errors';

export interface CameraApi {
  videoRef: React.RefObject<HTMLVideoElement>;
  active: boolean;
  starting: boolean;
  error: Explained | null;
  devices: MediaDeviceInfo[];
  deviceId: string;
  setDeviceId(id: string): void;
  start(): Promise<void>;
  stop(): void;
  /** Encodes the current frame (full camera resolution) as JPEG. Returns null if no frame is ready. */
  captureBlob(quality?: number): Promise<Blob | null>;
}

export function useCamera(): CameraApi {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [active, setActive] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<Explained | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState('');

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setActive(false);
  }, []);

  const start = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError({ title: 'Camera API unavailable', detail: 'This browser or context does not expose getUserMedia.', hint: 'Use a current browser over HTTPS or localhost.' });
      return;
    }
    setStarting(true);
    try {
      stop();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' }), width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      if (v) {
        v.srcObject = stream;
        await v.play();
      }
      setActive(true);
      setDevices((await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput'));
    } catch (e) {
      setError(explainError(e, 'Camera error'));
      stop();
    } finally {
      setStarting(false);
    }
  }, [deviceId, stop]);

  useEffect(() => stop, [stop]);

  const captureBlob = useCallback(async (quality = 0.92) => {
    const v = videoRef.current;
    if (!v || v.readyState < 2 || !v.videoWidth) return null;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d')?.drawImage(v, 0, 0);
    return new Promise<Blob | null>((res) => c.toBlob(res, 'image/jpeg', quality));
  }, []);

  return { videoRef, active, starting, error, devices, deviceId, setDeviceId, start, stop, captureBlob };
}
