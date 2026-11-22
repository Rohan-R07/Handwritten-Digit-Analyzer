import { Clock3, Trash2 } from 'lucide-react'
import type { RecentItem } from '../types/recognition'
export default function RecentRecognitions({ items, onClear }: { items: RecentItem[]; onClear: () => void }) {
  if (!items.length) return null
  return <section className="recent-section"><div className="section-title-row"><div><div className="section-eyebrow"><span>04</span> THIS SESSION</div><h2>Recent recognitions<span>.</span></h2></div><button className="clear-history" onClick={onClear}><Trash2 size={14}/> Clear history</button></div>
    <div className="recent-list">{items.map(item => <div className="recent-item" key={item.id}><img src={item.thumbnail} alt="Recent upload thumbnail"/><div className="recent-name"><b>{item.filename}</b><small>{item.status.toUpperCase()}</small></div><strong>{item.digit ?? '—'}</strong></div>)}</div>
  </section>
}