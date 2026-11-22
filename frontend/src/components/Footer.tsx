import { ScanLine } from 'lucide-react'
export default function Footer() {
  return <footer className="footer"><div className="footer-brand"><ScanLine size={18}/><b>DigitLens<span>.</span></b></div><p>Handwriting, decoded. A lightweight computer vision project.</p><div className="footer-stack">REACT <i/> OPENCV <i/> NUMPY</div><small>Built for learning. Powered by pixels.</small></footer>
}