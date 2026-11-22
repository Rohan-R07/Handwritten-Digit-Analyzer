import { useCallback, useState } from 'react'
import { recognizeDigit } from '../services/recognitionApi'
import type { RecognitionResult } from '../types/recognition'

export function useDigitRecognition() {
  const [result, setResult] = useState<RecognitionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const recognize = useCallback(async (file: File) => {
    if (loading) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const next = await recognizeDigit(file)
      setResult(next)
      return next
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Recognition failed.')
      return undefined
    } finally {
      setLoading(false)
    }
  }, [loading])

  const reset = useCallback(() => { setResult(null); setError(null) }, [])
  return { result, loading, error, recognize, reset }
}