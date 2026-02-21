export function usePagination(total: number, page: number, pageSize: number) {
  const totalPages = Math.ceil(total / pageSize)
  const hasNext = page < totalPages
  const hasPrev = page > 1
  return { totalPages, hasNext, hasPrev }
}
