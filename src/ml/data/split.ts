import { hashString, mulberry32, shuffleInPlace } from '@/utils/rng';

export interface SplitItem {
  id: string;
  classIndex: number;
  groupId: string;
  seq: number;
}

export interface SplitOptions {
  validationSplit: number;
  seed: number;
  /** Consecutive frames of one capture group are cut into blocks of this size. */
  blockSize?: number;
  /** Frames on the *train* side of a train/val block boundary that are dropped to avoid leakage. */
  purgeGap?: number;
}

export interface SplitResult {
  train: string[];
  val: string[];
  purged: string[];
  perClass: { classIndex: number; train: number; val: number; purged: number }[];
}

interface Unit {
  items: SplitItem[];
  groupId: string;
  block: number;
}

/**
 * Seeded, stratified, group-aware split.
 * - Every class is split independently so validation covers each class.
 * - Items sharing a groupId (a camera burst) are near-duplicates; they are cut into contiguous blocks
 *   and each block goes wholly to train or validation.
 * - Frames adjacent to a train/val boundary inside a burst are purged from training.
 * The result is independent of the input order.
 */
export function splitDataset(input: SplitItem[], opts: SplitOptions): SplitResult {
  const blockSize = opts.blockSize ?? 10;
  const purgeGap = opts.purgeGap ?? 2;
  const items = [...input].sort((a, b) => (a.id < b.id ? -1 : 1));
  const classIndices = [...new Set(items.map((i) => i.classIndex))].sort((a, b) => a - b);
  const result: SplitResult = { train: [], val: [], purged: [], perClass: [] };

  for (const ci of classIndices) {
    const rng = mulberry32(opts.seed ^ hashString(`class:${ci}`));
    const classItems = items.filter((i) => i.classIndex === ci);
    const groups = new Map<string, SplitItem[]>();
    for (const it of classItems) {
      const g = groups.get(it.groupId);
      if (g) g.push(it);
      else groups.set(it.groupId, [it]);
    }
    const units: Unit[] = [];
    for (const [groupId, members] of [...groups.entries()].sort(([a], [b]) => (a < b ? -1 : 1))) {
      members.sort((a, b) => a.seq - b.seq || (a.id < b.id ? -1 : 1));
      for (let i = 0, blk = 0; i < members.length; i += blockSize, blk++) {
        units.push({ items: members.slice(i, i + blockSize), groupId, block: blk });
      }
    }
    shuffleInPlace(units, rng);

    const n = classItems.length;
    const target = n >= 2 && opts.validationSplit > 0 ? Math.max(1, Math.round(n * opts.validationSplit)) : 0;
    const assignment = new Map<Unit, 'train' | 'val'>();
    let valCount = 0;
    let valUnits = 0;
    for (const u of units) {
      // Add a unit to validation only if that moves the class closer to the requested ratio (always at least one
      // unit when validation is requested), and always leave at least one unit for training.
      const len = u.items.length;
      const closer = valCount === 0 || Math.abs(valCount + len - target) < Math.abs(valCount - target);
      if (target > 0 && closer && valUnits < units.length - 1) {
        assignment.set(u, 'val');
        valCount += len;
        valUnits++;
      } else assignment.set(u, 'train');
    }

    // Purge train frames next to val frames of the same burst.
    const purged = new Set<string>();
    const byGroup = new Map<string, Unit[]>();
    for (const u of units) {
      const list = byGroup.get(u.groupId);
      if (list) list.push(u);
      else byGroup.set(u.groupId, [u]);
    }
    for (const list of byGroup.values()) {
      list.sort((a, b) => a.block - b.block);
      for (let k = 0; k < list.length - 1; k++) {
        const a = list[k] as Unit;
        const b = list[k + 1] as Unit;
        const sa = assignment.get(a);
        const sb = assignment.get(b);
        if (sa === sb) continue;
        const trainUnit = sa === 'train' ? a : b;
        const fromEnd = sa === 'train';
        const edge = fromEnd ? trainUnit.items.slice(-purgeGap) : trainUnit.items.slice(0, purgeGap);
        // Never purge a unit down to nothing.
        if (edge.length < trainUnit.items.length) edge.forEach((it) => purged.add(it.id));
      }
    }

    let tr = 0;
    let va = 0;
    for (const u of units) {
      const side = assignment.get(u);
      for (const it of u.items) {
        if (side === 'val') {
          result.val.push(it.id);
          va++;
        } else if (purged.has(it.id)) result.purged.push(it.id);
        else {
          result.train.push(it.id);
          tr++;
        }
      }
    }
    result.perClass.push({ classIndex: ci, train: tr, val: va, purged: classItems.length - tr - va });
  }
  return result;
}
