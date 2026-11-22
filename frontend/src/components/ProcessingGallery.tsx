import { useState } from 'react'
import { ImageOff } from 'lucide-react'
import type { ProcessingStage, RecognitionResult } from '../types/recognition'
const stages: { id: ProcessingStage; label: string; description: string }[] = [
  { id: 'original', label: 'Original', description: 'The image exactly as uploaded.' },
  { id: 'grayscale', label: 'Grayscale', description: 'Color information converted to intensity values.' },
  { id: 'thresholded', label: 'Thresholded', description: 'Foreground and background separated.' },
  { id: 'cleaned', label: 'Cleaned', description: 'Noise reduction from the processing pipeline.' },
  { id: 'cropped', label: 'Cropped', description: 'The detected digit region isolated.' },
  { id: 'normalized', label: 'Normalized 28×28', description: 'Digit resized to the model/template input dimensions.' },
]
export default function ProcessingGallery({ result }: { result: RecognitionResult | null }) {
  const [active, setActive] = useState<ProcessingStage>('original')
  if (!result) return null
  const available = stages.filter(stage => result.processingImages[stage.id])
  if (!available.length) return <section className="gallery-section"><div className="section-title-row"><div><div className="section-eyebrow"><span>03</span> PIPELINE VIEW</div><h2>How the computer sees your digit<span>.</span></h2></div></div><div className="empty-stage"><ImageOff size={22}/><p>Processing visualizations were not included in the backend response.</p><small>Return stage images from Python to display them here.</small></div></section>
  const selected = result.processingImages[active] || result.processingImages[available[0].id]
  const stage = stages.find(item => item.id === active && result.processingImages[item.id]) || available[0]
  return <section className="gallery-section"><div className="section-title-row"><div><div className="section-eyebrow"><span>03</span> PIPELINE VIEW</div><h2>How the computer sees your digit<span>.</span></h2></div><span className="stage-count">{available.length} / 6 STAGES RECEIVED</span></div>
    <div className="stage-tabs" role="tablist">{available.map(item => <button role="tab" aria-selected={stage.id === item.id} className={stage.id === item.id ? 'active' : ''} key={item.id} onClick={() => setActive(item.id)}>{item.label}</button>)}</div>
    <div className="stage-view"><div className="stage-image">{selected && <img src={selected.image} alt={`${stage.label} processing stage`}/>}</div><div className="stage-caption"><div><h3>{stage.label}</h3><p>{stage.description}</p></div><span>{selected?.width && selected?.height ? `${selected.width} × ${selected.height} px` : stage.id === 'normalized' ? '28 × 28 px' : 'IMAGE OUTPUT'}</span></div></div>
  </section>
}