/**
 * Apply jet colormap to a normalized value [0, 1].
 * Returns [r, g, b] in [0, 255].
 */
export function jetColormap(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)))
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)))
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)))
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}
