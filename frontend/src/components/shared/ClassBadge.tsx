interface Props {
  cls: string | null
  size?: 'sm' | 'md'
}

export function ClassBadge({ cls, size = 'md' }: Props) {
  const colors: Record<string, string> = {
    malignant: 'bg-red-900/60 text-red-300 border-red-700',
    benign: 'bg-green-900/60 text-green-300 border-green-700',
  }
  const pad = size === 'sm' ? 'px-1.5 py-0.5 text-xs' : 'px-2.5 py-1 text-sm'
  const style = colors[cls ?? ''] ?? 'bg-gray-800 text-gray-400 border-gray-700'
  return (
    <span className={`inline-flex items-center rounded-full border font-medium ${pad} ${style}`}>
      {cls ?? 'unknown'}
    </span>
  )
}
