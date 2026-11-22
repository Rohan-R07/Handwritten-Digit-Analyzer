import { AlertCircle, CheckCircle2, RotateCcw } from 'lucide-react'
import type { RecognitionResult as ResultType } from '../types/recognition'
import CandidateScores from './CandidateScores'
export default function RecognitionResult({ result, onRetry }: { result: ResultType; onRetry: () => void }) {
  const invalid = result.status === 'invalid'
  const uncertain = result.status === 'uncertain'
  return <section className={`result-panel ${invalid ? 'result-invalid' : uncertain ? 'result-uncertain' : ''}`} aria-live="polite">
    <div className="result-head"><div className="section-eyebrow"><span>02</span> RECOGNITION OUTPUT</div><span className={`status-pill ${invalid ? 'status-bad' : uncertain ? 'status-warn' : 'status-good'}`}>{invalid ? 'NO VALID DIGIT' : uncertain ? 'UNCERTAIN' : 'RECOGNIZED'}</span></div>
    <div className="result-main">
      <div className="digit-display">{invalid ? <AlertCircle size={48}/> : result.predictedDigit}</div>
      <div className="result-copy"><h3>{invalid ? 'No valid digit detected' : uncertain ? 'Best candidate found' : 'Your digit is decoded.'}</h3><p>{result.message || (invalid ? 'Try a clearer image containing one handwritten digit from 0 to 9.' : uncertain ? 'The engine returned a candidate, but the result is not conclusive.' : 'The recognition engine returned a digit prediction.')}</p></div>
    </div>
    <CandidateScores candidates={result.candidates}/>
    <button className="retry-button" onClick={onRetry}><RotateCcw size={15}/> TRY ANOTHER IMAGE</button>
  </section>
}