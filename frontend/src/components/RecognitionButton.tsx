import { ScanSearch } from 'lucide-react'
interface Props { onClick: () => void; disabled?: boolean; loading?: boolean }
export default function RecognitionButton({ onClick, disabled, loading }: Props) {
  return <button className="recognize-button" onClick={onClick} disabled={disabled || loading}>
    <span>{loading ? 'WORKING ON IT…' : 'RECOGNIZE DIGIT'}</span><ScanSearch size={19}/>
  </button>
}