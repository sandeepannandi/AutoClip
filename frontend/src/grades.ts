/**
 * CSS `filter` approximations of the ffmpeg grade presets in
 * `autoclip.pipeline.export.GRADES`.
 *
 * These are deliberately approximations — colortemperature and curves have no
 * CSS equivalent, so warm leans on sepia, punchy on saturation/contrast, and
 * the film fade on a lowered contrast with slight lift. They exist only so the
 * review preview reads 'warmer'/'cooler'/'punchier' the same way the export
 * does; the exported clip is always graded by libavfilter, never by this.
 * "none" and unknown keys map to undefined, which applies no filter at all.
 */
export const GRADE_FILTERS: Record<string, string> = {
  warm: 'sepia(0.24) saturate(1.12) contrast(1.03)',
  punchy: 'contrast(1.12) saturate(1.35)',
  cool: 'sepia(0.16) hue-rotate(18deg) saturate(1.06)',
  film: 'contrast(0.95) saturate(0.92) brightness(1.02)',
}

export function gradeFilter(grade: string | undefined | null): string | undefined {
  return grade && grade !== 'none' ? (GRADE_FILTERS[grade] ?? undefined) : undefined
}