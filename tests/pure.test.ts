import { describe, expect, it } from 'vitest';
import { splitDataset, type SplitItem } from '@/ml/data/split';
import { analyzeDataset } from '@/ml/data/preflight';
import { buildEvaluationReport } from '@/ml/metrics/classification';
import { assessQuality, lumaStats } from '@/ml/preprocessing/quality';
import { dHashFromLuma, hammingHex } from '@/ml/preprocessing/hash';
import { decide } from '@/ml/inference/decision';
import { affineParams } from '@/ml/augmentation/augment';
import type { DatasetClass, ImageSample } from '@/types/domain';

const items = (): SplitItem[] => {
  const out: SplitItem[] = [];
  // class 0: 60 frames from ONE camera burst; class 1: 30 independent uploads
  for (let i = 0; i < 60; i++) out.push({ id: `a${String(i).padStart(3, '0')}`, classIndex: 0, groupId: 'burst-1', seq: i });
  for (let i = 0; i < 30; i++) out.push({ id: `b${String(i).padStart(3, '0')}`, classIndex: 1, groupId: `up-${i}`, seq: 0 });
  return out;
};

describe('splitDataset', () => {
  it('is deterministic for a seed and independent of input order', () => {
    const a = splitDataset(items(), { validationSplit: 0.2, seed: 42 });
    const b = splitDataset([...items()].reverse(), { validationSplit: 0.2, seed: 42 });
    expect(b).toEqual(a);
    const c = splitDataset(items(), { validationSplit: 0.2, seed: 43 });
    expect(c.val).not.toEqual(a.val);
  });
  it('never overlaps train/val/purged and covers every item exactly once', () => {
    const r = splitDataset(items(), { validationSplit: 0.2, seed: 1 });
    const all = [...r.train, ...r.val, ...r.purged];
    expect(new Set(all).size).toBe(all.length);
    expect(all.length).toBe(90);
  });
  it('is stratified: every class gets validation samples near the requested ratio', () => {
    const r = splitDataset(items(), { validationSplit: 0.2, seed: 1 });
    for (const pc of r.perClass) expect(pc.val).toBeGreaterThan(0);
    expect(r.val.length).toBeGreaterThanOrEqual(15);
    expect(r.val.length).toBeLessThanOrEqual(24);
  });
  it('keeps burst frames in contiguous blocks and purges train frames next to validation blocks', () => {
    const r = splitDataset(items(), { validationSplit: 0.2, seed: 5, blockSize: 10, purgeGap: 2 });
    const val = new Set(r.val);
    const train = new Set(r.train);
    for (let i = 0; i < 60; i++) {
      const id = `a${String(i).padStart(3, '0')}`;
      if (!train.has(id)) continue;
      // no train frame within 2 frames of a validation frame of the same burst
      for (let d = -2; d <= 2; d++) {
        const nb = `a${String(i + d).padStart(3, '0')}`;
        expect(val.has(nb)).toBe(false);
      }
    }
    expect(r.purged.length).toBeGreaterThan(0);
  });
  it('keeps at least one training unit even for tiny classes', () => {
    const r = splitDataset([{ id: 'x1', classIndex: 0, groupId: 'g', seq: 0 }, { id: 'x2', classIndex: 0, groupId: 'h', seq: 0 }], { validationSplit: 0.9, seed: 1 });
    expect(r.train.length).toBeGreaterThanOrEqual(1);
  });
});

describe('classification metrics', () => {
  it('matches a hand-computed confusion matrix, precision, recall and F1', () => {
    // actual: 0 0 0 0 1 1 1 2 2 2   predicted: 0 0 1 0 1 1 2 2 2 0
    const actual = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2];
    const pred = [0, 0, 1, 0, 1, 1, 2, 2, 2, 0];
    const probs = new Float32Array(30);
    pred.forEach((p, i) => (probs[i * 3 + p] = 0.9));
    const r = buildEvaluationReport({ classIds: ['a', 'b', 'c'], classNames: ['A', 'B', 'C'], sampleIds: actual.map((_, i) => `s${i}`), yTrue: actual, probs });
    expect(r.confusion).toEqual([[3, 1, 0], [0, 2, 1], [1, 0, 2]]);
    expect(r.accuracy).toBeCloseTo(0.7);
    expect(r.perClass[0]?.precision).toBeCloseTo(3 / 4);
    expect(r.perClass[0]?.recall).toBeCloseTo(3 / 4);
    expect(r.perClass[1]?.precision).toBeCloseTo(2 / 3);
    expect(r.perClass[1]?.recall).toBeCloseTo(2 / 3);
    expect(r.perClass[2]?.f1).toBeCloseTo(2 / 3);
    expect(r.macroF1).toBeCloseTo((0.75 + 2 / 3 + 2 / 3) / 3);
    expect(r.misclassified).toHaveLength(3);
  });
});

describe('image quality', () => {
  const flat = (v: number) => new Uint8Array(64 * 64).fill(v);
  it('rejects blank, dark and bright frames, accepts a textured one', () => {
    expect(assessQuality(lumaStats(flat(128), 64), 500, 500).reject?.reason).toBe('blank');
    const noise = Uint8Array.from({ length: 4096 }, (_, i) => (i * 7919) % 21);
    expect(assessQuality(lumaStats(noise, 64), 500, 500).reject?.reason).toBe('too-dark');
    const bright = Uint8Array.from({ length: 4096 }, (_, i) => 244 + ((i * 7919) % 12));
    expect(assessQuality(lumaStats(bright, 64), 500, 500).reject?.reason).toBe('too-bright');
    const tex = Uint8Array.from({ length: 4096 }, (_, i) => ((i * 2654435761) >>> 24) & 255);
    const ok = assessQuality(lumaStats(tex, 64), 500, 500);
    expect(ok.reject).toBeNull();
    expect(ok.report.warnings).not.toContain('blurry');
  });
  it('rejects tiny images and warns on low resolution', () => {
    const tex = Uint8Array.from({ length: 4096 }, (_, i) => ((i * 2654435761) >>> 24) & 255);
    expect(assessQuality(lumaStats(tex, 64), 40, 400).reject?.reason).toBe('too-small');
    expect(assessQuality(lumaStats(tex, 64), 100, 400).report.warnings).toContain('low-resolution');
  });
  it('flags smooth gradients as blurry', () => {
    const grad = Uint8Array.from({ length: 4096 }, (_, i) => 40 + ((i % 64) * 2));
    expect(assessQuality(lumaStats(grad, 64), 500, 500).report.warnings).toContain('blurry');
  });
});

describe('hashing', () => {
  it('hamming distance of dHash', () => {
    const a = Array.from({ length: 72 }, (_, i) => (i * 37) % 200);
    const b = a.map((v, i) => (i === 3 ? v + 5 : v));
    expect(hammingHex(dHashFromLuma(a), dHashFromLuma(a))).toBe(0);
    expect(hammingHex(dHashFromLuma(a), dHashFromLuma(b))).toBeLessThanOrEqual(2);
    expect(hammingHex(dHashFromLuma(a), dHashFromLuma(a.map((v) => 255 - v)))).toBeGreaterThan(20);
  });
});

describe('decision threshold', () => {
  const p = (a: number, b: number, c: number) => [
    { classId: '1', className: 'Cat', probability: a }, { classId: '2', className: 'Dog', probability: b }, { classId: '3', className: 'Bird', probability: c },
  ];
  it('accepts confident and rejects diffuse predictions', () => {
    expect(decide(p(0.962, 0.028, 0.01), 0.7).label).toBe('Cat');
    const d = decide(p(0.42, 0.36, 0.22), 0.7);
    expect(d.isUnknown).toBe(true);
    expect(d.label).toBeNull();
    expect(d.entropy).toBeGreaterThan(0.9);
  });
});

describe('affine parameters', () => {
  it('identity for theta=0, zoom=1, t=0', () => {
    const p = affineParams(64, 64, 0, 1, 0, 0);
    expect(p.map((v) => Math.round(v * 1e6) / 1e6 + 0)).toEqual([1, 0, 0, 0, 1, 0, 0, 0]);
  });
});

describe('preflight', () => {
  const classes: DatasetClass[] = [{ id: 'c1', name: 'Cat', color: '#fff', createdAt: 0 }, { id: 'c2', name: 'Dog', color: '#fff', createdAt: 0 }];
  const mk = (classId: string, n: number, source: 'upload' | 'camera' = 'upload', group = ''): ImageSample[] =>
    Array.from({ length: n }, (_, i) => ({
      id: `${classId}-${i}`, projectId: 'p', classId, source, name: `${classId}-${i}.jpg`, width: 300, height: 300, mimeType: 'image/jpeg', size: 1, createdAt: 0,
      groupId: group || `${classId}-${i}`, seq: i, sha256: 'abcdef012345', phash: '0', quality: { meanLuma: 100, stdLuma: 40, sharpness: 100, warnings: [] },
    }));
  it('reports class imbalance and requires acknowledgement', () => {
    const a = analyzeDataset(classes, [...mk('c1', 120), ...mk('c2', 31)], { mode: 'transfer', validationSplit: 0.2 });
    expect(a.blocking).toBe(false);
    expect(a.needsAcknowledgement).toBe(true);
    expect(a.issues.find((i) => i.code === 'imbalance')?.message).toContain('Cat: 120');
  });
  it('blocks when a class has too few samples or only one class exists', () => {
    expect(analyzeDataset(classes, [...mk('c1', 50), ...mk('c2', 3)], { mode: 'transfer', validationSplit: 0.2 }).blocking).toBe(true);
    expect(analyzeDataset(classes, mk('c1', 50), { mode: 'transfer', validationSplit: 0.2 }).blocking).toBe(true);
  });
  it('warns when a class comes from a single camera burst', () => {
    const a = analyzeDataset(classes, [...mk('c1', 60, 'camera', 'burst'), ...mk('c2', 60)], { mode: 'transfer', validationSplit: 0.2 });
    expect(a.issues.some((i) => i.code === 'single-burst')).toBe(true);
  });
});
