import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { ValidationBrowser } from './pages/ValidationBrowser'
import { KernelExplorer } from './pages/KernelExplorer'
import { TrainingPage } from './pages/TrainingPage'

export function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<ValidationBrowser />} />
          <Route path="/kernels" element={<KernelExplorer />} />
          <Route path="/training" element={<TrainingPage />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  )
}
