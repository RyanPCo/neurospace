import { ImageSummary } from '../../types'
import { ClassBadge } from '../shared/ClassBadge'
import { imagesApi } from '../../api/images'

interface Props {
  image: ImageSummary
  selected: boolean
  onClick: () => void
}

export function ImageCard({ image, selected, onClick }: Props) {
  return (
    <div
      onClick={onClick}
      className={`cursor-pointer rounded-lg overflow-hidden border transition-all ${
        selected
          ? 'border-brand-500 ring-2 ring-brand-500/30'
          : 'border-gray-800 hover:border-gray-600'
      } bg-gray-900`}
    >
      <div className="aspect-video relative bg-gray-800">
        <img
          src={imagesApi.fileUrl(image.id)}
          alt={image.filename}
          className="w-full h-full object-cover"
          loading="lazy"
        />
        {image.annotation_count > 0 && (
          <div className="absolute top-1.5 right-1.5 bg-blue-600 text-white text-xs px-1.5 py-0.5 rounded-full">
            {image.annotation_count}
          </div>
        )}
      </div>
      <div className="p-2 space-y-1">
        <div className="flex items-center justify-between gap-1">
          <ClassBadge cls={image.predicted_class} size="sm" />
          <span className="text-xs text-gray-500 capitalize">{image.split}</span>
        </div>
        <div className="text-xs text-gray-500 truncate capitalize">GT: {image.ground_truth}</div>
      </div>
    </div>
  )
}
