import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { ValidationBrowser } from './pages/ValidationBrowser'
import { KernelExplorer } from './pages/KernelExplorer'
import { TrainingPage } from './pages/TrainingPage'
import { VolumeViewerPage } from './pages/VolumeViewerPage'
import { AnnotationPopupPage } from './pages/AnnotationPopupPage'
import { Toaster } from './components/shared/Toast'

export function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<ValidationBrowser />} />
          <Route path="/kernels" element={<KernelExplorer />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/viewer3d" element={<VolumeViewerPage />} />
          <Route path="/annotate/:imageId" element={<AnnotationPopupPage />} />
        </Routes>
      </AppShell>
      <Toaster />
    </BrowserRouter>
  )
}
