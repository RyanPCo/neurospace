import { useImageStore } from '../../store/imageStore'

const MAGS = ['40X', '100X', '200X', '400X']

export function MagnificationFilter() {
  const { filters, setFilters } = useImageStore()

  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500">Magnification:</span>
      <div className="flex gap-1">
        <button
          onClick={() => setFilters({ magnification: undefined })}
          className={`px-2.5 py-1 rounded text-xs transition-colors ${
            !filters.magnification ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          All
        </button>
        {MAGS.map(m => (
          <button
            key={m}
            onClick={() => setFilters({ magnification: filters.magnification === m ? undefined : m })}
            className={`px-2.5 py-1 rounded text-xs transition-colors ${
              filters.magnification === m ? 'bg-brand-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {m}
          </button>
        ))}
      </div>
    </div>
  )
}
