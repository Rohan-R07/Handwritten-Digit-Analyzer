import { CheckCircle2, Circle, LoaderCircle } from 'lucide-react'
const steps = ['Analyzing handwriting', 'Extracting digit features', 'Comparing reference templates']
export default function ProcessingState() {
  return <section className="processing-box" aria-live="polite"><div className="processing-top"><LoaderCircle className="spin" size={26}/><div><h3>Analyzing handwriting<span className="ellipsis">...</span></h3><p>Your image is moving through the recognition pipeline.</p></div></div>
    <div className="processing-steps">{steps.map((step, i) => <div className="processing-step" key={step}>{i === 0 ? <CheckCircle2 size={16}/> : <Circle size={16}/>}<span>{step}</span></div>)}</div>
  </section>
}