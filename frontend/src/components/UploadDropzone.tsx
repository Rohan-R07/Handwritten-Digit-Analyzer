import { useRef, useState, type DragEvent, type ChangeEvent } from 'react'
import { FileImage, Upload, X } from 'lucide-react'

interface Props { file: File | null; onFile: (file: File) => void; onRemove: () => void; disabled?: boolean }
const MAX_SIZE = 10 * 1024 * 1024
function validate(file: File): string | null {
  if (!['image/png', 'image/jpeg'].includes(file.type)) return 'Unsupported format. Please choose a PNG, JPG or JPEG image.'
  if (file.size > MAX_SIZE) return 'This image is larger than 10 MB. Please choose a smaller file.'
  return null
}
export default function UploadDropzone({ file, onFile, onRemove, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const accept = (next: File | undefined) => {
    if (!next) return
    const issue = validate(next)
    setError(issue || '')
    if (!issue) onFile(next)
  }
  const onChange = (e: ChangeEvent<HTMLInputElement>) => { accept(e.target.files?.[0]); e.target.value = '' }
  const onDrop = (e: DragEvent<HTMLDivElement>) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files?.[0]) }
  return <div className="upload-wrap">
    <div className={`dropzone ${dragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`}
      onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
      <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,image/png,image/jpeg" onChange={onChange} hidden />
      <div className="upload-icon"><Upload size={25} strokeWidth={2.5} /></div>
      <h3>{file ? 'Your image is ready' : 'Drop your handwritten digit here'}</h3>
      <p>{file ? file.name : <>or <button className="text-button" onClick={() => inputRef.current?.click()} disabled={disabled}>browse files</button></>}</p>
      <span className="upload-meta"><FileImage size={14} /> PNG, JPG OR JPEG <b>·</b> ONE DIGIT PER IMAGE <b>·</b> MAX 10 MB</span>
      {file && <button className="remove-file" onClick={onRemove} disabled={disabled}><X size={15}/> Remove image</button>}
      {!file && <button className="browse-button" onClick={() => inputRef.current?.click()} disabled={disabled}>CHOOSE AN IMAGE</button>}
    </div>
    <div className="upload-hint"><span className="hint-mark">!</span><span>For best results, upload one clear, conventional handwritten digit from <b>0 to 9</b> on a plain background.</span></div>
    {error && <p className="inline-error" role="alert">{error}</p>}
  </div>
}