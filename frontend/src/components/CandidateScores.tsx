import type { Candidate } from '../types/recognition'
export default function CandidateScores({ candidates }: { candidates: Candidate[] }) {
  if (!candidates.length) return null
  return <div className="candidate-area"><div className="mini-heading">ALTERNATIVE CANDIDATES</div><div className="candidate-list">{candidates.slice(0, 5).map((item, i) => <div className="candidate" key={`${item.digit}-${i}`}><span className="candidate-digit">{item.digit}</span><span className="candidate-score">{item.score !== undefined ? `Score: ${item.score}` : 'Candidate'}</span></div>)}</div></div>
}