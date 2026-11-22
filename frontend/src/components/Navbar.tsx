import { Github, ScanLine } from 'lucide-react'

export default function Navbar() {
  return <header className="nav-shell">
    <a className="brand" href="#" aria-label="DigitLens home"><span className="brand-icon"><ScanLine size={21} strokeWidth={2.6} /></span><span>DigitLens<span className="brand-period">.</span></span></a>
    <div className="nav-right"><span className="nav-chip"><span className="live-dot" /> COMPUTER VISION</span><a className="github-link" href="https://github.com/" target="_blank" rel="noreferrer" aria-label="GitHub"><Github size={18} /></a></div>
  </header>
}