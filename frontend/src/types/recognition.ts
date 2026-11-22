export type RecognitionStatus = 'success' | 'uncertain' | 'invalid'

export interface Candidate {
  digit: number | string
  score?: number | string
  [key: string]: unknown
}

export interface ProcessingImage {
  image: string
  width?: number
  height?: number
}

export type ProcessingStage = 'original' | 'grayscale' | 'thresholded' | 'cleaned' | 'cropped' | 'normalized'

export interface RecognitionResult {
  status: RecognitionStatus
  predictedDigit?: number | string
  message?: string
  candidates: Candidate[]
  processingImages: Partial<Record<ProcessingStage, ProcessingImage>>
  details: Record<string, unknown>
  raw: unknown
}

export interface RecentItem {
  id: string
  digit: number | string | null
  status: RecognitionStatus
  filename: string
  thumbnail: string
}