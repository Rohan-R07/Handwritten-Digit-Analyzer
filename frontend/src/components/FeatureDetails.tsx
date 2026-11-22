import { ChevronDown, Code2 } from 'lucide-react'
import type { RecognitionResult } from '../types/recognition'
function pretty(value: unknown): string {
  if (typeof value === 'object' && value !== null) return JSON.stringify(value, null, 2)
  return String(value)
}
export default function FeatureDetails({ result }: { result: RecognitionResult | null }) {
  if (!result || !Object.keys(result.details).length) return null
  return <details className="details-box"><summary><span className="details-title"><Code2 size={18}/> UNDER THE HOOD</span><span className="details-subtitle">Recognition details <ChevronDown size={16}/></span></summary>
    <div className="details-content"><p>Technical values returned by the recognition engine.</p><div className="details-grid">{Object.entries(result.details).map(([key, value]) => <div className="detail-item" key={key}><span>{key.replaceAll('_', ' ')}</span><pre>{pretty(value)}</pre></div>)}</div></div>
  </details>
}