/**
 * Single source of truth for colors used outside Tailwind's class system — inline SVG strokes/fills,
 * canvas-free chart bars, dynamic class-color swatches. Mirrors the token values in tailwind.config.js
 * so the two never drift apart. Update both together when changing the theme.
 */
export const PALETTE = {
  bg: '#f3f4f7',
  surface: '#ffffff',
  raised: '#eef0f4',
  line: '#d7dbe3',
  ink: '#14161c',
  mute: '#586071',
  faint: '#8890a0',
  accent: '#3457e6',
  accentStrong: '#2743e6',
  good: '#157f4a',
  warn: '#a15a00',
  bad: '#c92a3e',
  sky: '#0969da',
  violet: '#7c4fd6',
} as const;

export const CLASS_COLORS = [PALETTE.accent, PALETTE.good, PALETTE.warn, PALETTE.bad, PALETTE.sky, PALETTE.violet, '#0f9488', '#c2540a'];
