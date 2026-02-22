import { useImageStore } from '../../store/imageStore'

const SPLITS = ['train', 'val', 'test']

export function MagnificationFilter() {
  const { filters, setFilters } = useImageStore()

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500">Split:</span>
      <div className="flex gap-1">
        <button
          onClick={() => setFilters({ split: undefined })}
          className={`px-2.5 py-1 rounded text-xs transition-colors ${
            !filters.split ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          All
        </button>
        {SPLITS.map(split => (
          <button
            key={split}
            onClick={() => setFilters({ split: filters.split === split ? undefined : split })}
            className={`px-2.5 py-1 rounded text-xs transition-colors ${
              filters.split === split ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {split}
          </button>
        ))}
      </div>
    </div>
  )
}
