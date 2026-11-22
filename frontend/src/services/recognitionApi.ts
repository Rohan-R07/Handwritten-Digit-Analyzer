import axios from 'axios'
import type { Candidate, ProcessingImage, ProcessingStage, RecognitionResult } from '../types/recognition'

const baseURL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '')
const timeout = Number(import.meta.env.VITE_API_TIMEOUT_MS || 30000)

const api = axios.create({ baseURL, timeout })

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await api.get('/health')
    return response.status >= 200 && response.status < 300
  } catch {
    return false
  }
}

function toImage(value: unknown): ProcessingImage | undefined {
  if (typeof value === 'string') {
    const image = value.startsWith('data:') ? value : `data:image/png;base64,${value}`
    return { image }
  }
  if (!value || typeof value !== 'object') return undefined
  const obj = value as Record<string, unknown>
  const raw = obj.image ?? obj.url ?? obj.data ?? obj.base64 ?? obj.src
  if (typeof raw !== 'string') return undefined
  const image = raw.startsWith('data:') || raw.startsWith('http') || raw.startsWith('blob:')
    ? raw : `data:image/png;base64,${raw}`
  return {
    image,
    width: typeof obj.width === 'number' ? obj.width : undefined,
    height: typeof obj.height === 'number' ? obj.height : undefined,
  }
}

function normalizeResponse(raw: unknown): RecognitionResult {
  if (!raw || typeof raw !== 'object') throw new Error('The recognition service returned an empty or malformed response.')
  const data = raw as Record<string, unknown>
  const digitValue = data.predicted_digit ?? data.predictedDigit ?? data.digit ?? data.prediction
  const validDigit = typeof digitValue === 'number' || (typeof digitValue === 'string' && /^[0-9]$/.test(digitValue))
  const rawStatus = String(data.status ?? data.recognition_status ?? '').toLowerCase()
  let status: RecognitionResult['status'] = 'success'
  if (['invalid', 'no_digit', 'not_detected', 'failed'].includes(rawStatus) || data.valid === false || data.is_valid === false) status = 'invalid'
  else if (['uncertain', 'low_confidence', 'ambiguous'].includes(rawStatus)) status = 'uncertain'
  else if (!validDigit) throw new Error('The backend response did not include a valid predicted digit or an invalid-input status.')

  const candidateValue = data.candidates ?? data.alternatives ?? data.rankings
  const candidates: Candidate[] = Array.isArray(candidateValue)
    ? candidateValue.filter((item): item is Candidate => !!item && typeof item === 'object' && 'digit' in item)
    : []
  const stageRoot = (data.processing_images ?? data.processingImages ?? data.stages ?? {}) as Record<string, unknown>
  const stages: ProcessingStage[] = ['original', 'grayscale', 'thresholded', 'cleaned', 'cropped', 'normalized']
  const processingImages: RecognitionResult['processingImages'] = {}
  for (const stage of stages) {
    const found = toImage(stageRoot[stage] ?? data[stage])
    if (found) processingImages[stage] = found
  }
  const details = (data.details ?? data.features ?? data.recognition_details ?? {}) as Record<string, unknown>
  return {
    status,
    predictedDigit: validDigit ? digitValue as number | string : undefined,
    message: typeof data.message === 'string' ? data.message : typeof data.reason === 'string' ? data.reason : undefined,
    candidates,
    processingImages,
    details: details && typeof details === 'object' ? details : {},
    raw,
  }
}

export async function recognizeDigit(file: File): Promise<RecognitionResult> {
  const form = new FormData()
  form.append('file', file)
  try {
    const response = await api.post('/api/recognize', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return normalizeResponse(response.data)
  } catch (error) {
    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNABORTED') throw new Error('The request timed out. Please try again.')
      if (!error.response) throw new Error('Backend unavailable. Check that your FastAPI server is running and the API URL is correct.')
      const detail = (error.response.data as { detail?: string; message?: string } | undefined)
      throw new Error(detail?.detail || detail?.message || `Recognition failed (HTTP ${error.response.status}).`)
    }
    throw error instanceof Error ? error : new Error('Something went wrong while recognizing the image.')
  }
}