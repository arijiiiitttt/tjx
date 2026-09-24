import type { PreprocessingSpec } from '@/types/ml';
import type { RawSample } from '@/types/worker';
import { bitmapToPixels, decodeBlob } from '../preprocessing/decode';

export interface PreparedSample {
  id: string;
  classIndex: number;
  groupId: string;
  seq: number;
  pixels: Uint8Array;
}

/** Decode blobs -> fixed-size RGB bytes. Bitmaps are closed immediately; failures are reported, not thrown. */
export async function decodeSamples(
  raw: RawSample[],
  classIndexById: Map<string, number>,
  spec: PreprocessingSpec,
  onProgress?: (done: number, total: number) => void,
  isAborted?: () => boolean,
): Promise<{ prepared: PreparedSample[]; failed: string[] }> {
  const prepared: PreparedSample[] = [];
  const failed: string[] = [];
  for (let i = 0; i < raw.length; i++) {
    if (isAborted?.()) break;
    const r = raw[i] as RawSample;
    const classIndex = classIndexById.get(r.classId);
    if (classIndex === undefined) continue;
    try {
      const bmp = await decodeBlob(r.blob);
      try {
        prepared.push({ id: r.id, classIndex, groupId: r.groupId, seq: r.seq, pixels: bitmapToPixels(bmp, spec) });
      } finally {
        bmp.close();
      }
    } catch {
      failed.push(r.id);
    }
    if (i % 8 === 7 || i === raw.length - 1) onProgress?.(i + 1, raw.length);
  }
  return { prepared, failed };
}
