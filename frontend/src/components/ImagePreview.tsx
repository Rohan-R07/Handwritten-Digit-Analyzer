import { Image as ImageIcon, RefreshCw, Trash2 } from 'lucide-react'
import { useEffect, useState } from 'react'

interface Props { file: File; onRemove: () => void; onReplace: () => void; disabled?: boolean }
export default function ImagePreview({ file, onRemove, onReplace, disabled }: Props) {
  const [url, setUrl] = useState('')
  const [dimensions, setDimensions] = useState('')
  useEffect(() => {
    const objectUrl = URL.createObjectURL(file)
    setUrl(objectUrl)
    const img = new Image()
    img.onload = () => setDimensions(`${img.naturalWidth} × ${img.naturalHeight} px`)
    img.src = objectUrl
    return () => URL.revokeObjectURL(objectUrl)
  }, [file])
  return <section className="preview-panel">
    <div className="section-eyebrow"><span>01</span> SELECTED IMAGE</div>
    <div className="preview-content">
      <div className="preview-image-frame">{url && <img src={url} alt="Selected handwritten digit preview" />}</div>
      <div className="preview-info"><div className="file-label"><ImageIcon size={16}/><span>IMAGE FILE</span></div><h3>{file.name}</h3><p>{dimensions || 'Reading dimensions…'} <b>·</b> {(file.size / 1024).toFixed(1)} KB</p>
        <div className="preview-actions"><button className="small-action" onClick={onReplace} disabled={disabled}><RefreshCw size={14}/> Replace</button><button className="small-action danger-action" onClick={onRemove} disabled={disabled}><Trash2 size={14}/> Remove</button></div>
      </div>
    </div>
  </section>
}