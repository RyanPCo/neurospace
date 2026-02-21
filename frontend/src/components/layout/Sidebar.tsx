import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Validation Browser', icon: '🔬' },
  { to: '/kernels', label: 'Kernel Explorer', icon: '🧩' },
  { to: '/training', label: 'Training', icon: '📈' },
]

export function Sidebar() {
  return (
    <aside className="w-52 bg-gray-900 border-r border-gray-800 flex flex-col">
      <div className="px-4 py-5 border-b border-gray-800">
        <span className="font-bold text-lg text-brand-500">CancerScope</span>
        <div className="text-xs text-gray-500 mt-0.5">Histopathology AI</div>
      </div>
      <nav className="flex-1 py-4 space-y-1 px-2">
        {links.map(l => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-brand-600 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-gray-100'
              }`
            }
          >
            <span>{l.icon}</span>
            {l.label}
          </NavLink>
        ))}
      </nav>
      <div className="px-4 py-3 border-t border-gray-800 text-xs text-gray-600">
        v1.0.0 · localhost
      </div>
    </aside>
  )
}
