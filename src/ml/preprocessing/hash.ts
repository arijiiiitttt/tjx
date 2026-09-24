export async function sha256Hex(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

/** dHash from a (size+1) x size grayscale buffer, 64 bits as 16 hex chars. */
export function dHashFromLuma(luma: ArrayLike<number>, w = 9, h = 8): string {
  let bits = '';
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w - 1; x++) {
      bits += (luma[y * w + x] as number) > (luma[y * w + x + 1] as number) ? '1' : '0';
    }
  }
  let hex = '';
  for (let i = 0; i < bits.length; i += 4) hex += parseInt(bits.slice(i, i + 4), 2).toString(16);
  return hex;
}

export function hammingHex(a: string, b: string): number {
  let d = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    let x = parseInt(a[i] as string, 16) ^ parseInt(b[i] as string, 16);
    while (x) {
      d += x & 1;
      x >>= 1;
    }
  }
  return d + Math.abs(a.length - b.length) * 4;
}
