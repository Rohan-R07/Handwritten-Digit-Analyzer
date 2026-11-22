import { ArrowDownRight, Sparkles } from 'lucide-react'

export default function HeroSection() {
  return <section className="hero">
    <div className="hero-kicker"><Sparkles size={14} /> A TINY LAB FOR BIG IDEAS</div>
    <div className="hero-grid">
      <div><h1>Handwriting,<br /><span>decoded.</span></h1></div>
      <div className="hero-copy"><p>Turn a handwritten digit into a digital prediction. Upload an image and explore how computer vision processes and recognizes it.</p><div className="tech-line"><span>OPENCV</span><i /> <span>NUMPY</span><i /> <span className="tech-badge">RULE-BASED RECOGNITION</span></div><a href="#workspace" className="jump-link">GO TO THE WORKSPACE <ArrowDownRight size={16} /></a></div>
    </div>
    <div className="hero-stamp" aria-hidden="true"><span>0—9</span><small>DIGIT<br/>DETECTOR</small></div>
  </section>
}