interface Props {
  cls: string | null
  size?: 'sm' | 'md'
}

const LABELS: Record<string, string> = {
  gradcam_focus: '⚡ Focus',
  notumor: 'No Tumor',
}

const COLORS: Record<string, string> = {
  glioma:        'bg-red-600 text-white',
  meningioma:    'bg-orange-500 text-white',
  pituitary:     'bg-blue-600 text-white',
  notumor:       'bg-green-600 text-white',
  gradcam_focus: 'bg-amber-500 text-black',
}

export function ClassBadge({ cls, size = 'md' }: Props) {
  const pad   = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-sm'
  const style = COLORS[cls ?? ''] ?? 'bg-gray-700 text-gray-300'
  return (
    <span className={`inline-flex items-center rounded-full font-medium ${pad} ${style}`}>
      {LABELS[cls ?? ''] ?? cls ?? 'unknown'}
    </span>
  )
}
