import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Activity } from 'lucide-react'
import Navbar from './components/Navbar'
import HeroSection from './components/HeroSection'
import UploadDropzone from './components/UploadDropzone'
import ImagePreview from './components/ImagePreview'
import RecognitionButton from './components/RecognitionButton'
import ProcessingState from './components/ProcessingState'
import RecognitionResult from './components/RecognitionResult'
import ProcessingGallery from './components/ProcessingGallery'
import FeatureDetails from './components/FeatureDetails'
import RecentRecognitions from './components/RecentRecognitions'
import Footer from './components/Footer'
import { useDigitRecognition } from './hooks/useDigitRecognition'
import { checkHealth } from './services/recognitionApi'
import type { RecentItem } from './types/recognition'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [history, setHistory] = useState<RecentItem[]>([])
  const [apiState, setApiState] = useState<'checking' | 'online' | 'offline'>('checking')
  const inputRef = useRef<HTMLInputElement>(null)
  const { result, loading, error, recognize, reset } = useDigitRecognition()

  useEffect(() => {
    let mounted = true
    checkHealth().then(ok => { if (mounted) setApiState(ok ? 'online' : 'offline') })
    return () => { mounted = false }
  }, [])

  const chooseFile = (next: File) => { setFile(next); reset() }
  const removeFile = () => { setFile(null); reset() }
  const replaceFile = () => inputRef.current?.click()
  const runRecognition = async () => {
    if (!file || loading) return
    const output = await recognize(file)
    if (output) {
      const item: RecentItem = {
        id: `${Date.now()}-${Math.random()}`,
        digit: output.predictedDigit ?? null,
        status: output.status,
        filename: file.name,
        thumbnail: URL.createObjectURL(file),
      }
      setHistory(previous => [item, ...previous].slice(0, 5))
    }
  }
  const retry = () => { reset(); setFile(null) }

  return <div className="app-shell">
    <Navbar/>
    <main>
      <HeroSection/>
      <section className="workspace-section" id="workspace">
        <div className="workspace-heading"><div><div className="section-eyebrow"><span>01</span> THE WORKSPACE</div><h2>Let’s read some<br className="mobile-break"/> handwriting<span>.</span></h2></div>
          <div className={`api-indicator ${apiState}`}><span className="api-dot"/>{apiState === 'checking' ? 'CHECKING ENGINE' : apiState === 'online' ? 'ENGINE ONLINE' : 'ENGINE OFFLINE'}</div>
        </div>
        {apiState === 'offline' && <div className="backend-notice"><AlertTriangle size={18}/><div><b>Backend not connected</b><p>Start your FastAPI server and check <code>VITE_API_BASE_URL</code>. You can still select and preview an image.</p></div></div>}
        <UploadDropzone file={file} onFile={chooseFile} onRemove={removeFile} disabled={loading}/>
        {file && <><ImagePreview file={file} onRemove={removeFile} onReplace={replaceFile} disabled={loading}/>
          <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,image/png,image/jpeg" hidden onChange={e => { const next = e.target.files?.[0]; if (next) chooseFile(next); e.target.value = '' }}/>
          <div className="recognize-row"><div className="recognize-note"><Activity size={16}/> Recognition starts only when you click.</div><RecognitionButton onClick={runRecognition} disabled={!file || loading} loading={loading}/></div>
        </>}
        {loading && <ProcessingState/>}
        {error && <div className="error-panel" role="alert"><AlertTriangle size={19}/><div><b>We couldn’t complete recognition</b><p>{error}</p><button onClick={runRecognition}>Try again</button></div></div>}
        {result && <RecognitionResult result={result} onRetry={retry}/>}
      </section>
      <div className="lower-content">
        <ProcessingGallery result={result}/>
        <FeatureDetails result={result}/>
        <RecentRecognitions items={history} onClear={() => { history.forEach(item => URL.revokeObjectURL(item.thumbnail)); setHistory([]) }}/>
      </div>
    </main>
    <Footer/>
  </div>
}