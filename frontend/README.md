# DigitLens Frontend

A responsive React + Vite + TypeScript frontend for your existing Python/OpenCV handwritten digit recognition engine. The visual direction is **neo-brutalist**: warm paper background, thick ink borders, offset shadows, bold typography and punchy violet/lime accents.

## Requirements
- Node.js 18.18+ (or Node 20+)
- Your FastAPI backend running separately

## Start the frontend
```bash
cd digitlens-frontend
npm install
cp .env.example .env
npm run dev
```
Open the local URL printed by Vite (usually http://localhost:5173).

Set `VITE_API_BASE_URL` in `.env` to your FastAPI server. Restart Vite after changing environment variables.

## Expected backend endpoints
- `GET /health`
- `POST /api/recognize` with multipart field named `file`

The frontend does **not** implement recognition logic and does not generate sample predictions. It displays only data returned by your backend.

### Suggested response shape
The UI accepts common response key variants, but this is the clearest contract:
```json
{
  "status": "success",
  "predicted_digit": 7,
  "message": "Digit recognized",
  "candidates": [{"digit": 7, "score": 0.91}, {"digit": 1, "score": 0.42}],
  "processing_images": {
    "original": {"image": "data:image/png;base64,...", "width": 640, "height": 480},
    "grayscale": {"image": "data:image/png;base64,...", "width": 640, "height": 480},
    "thresholded": {"image": "data:image/png;base64,..."},
    "cleaned": {"image": "data:image/png;base64,..."},
    "cropped": {"image": "data:image/png;base64,..."},
    "normalized": {"image": "data:image/png;base64,...", "width": 28, "height": 28}
  },
  "details": {
    "template_scores": {"7": 0.91, "1": 0.42},
    "aspect_ratio": 0.62,
    "foreground_density": 0.18,
    "hole_count": 0,
    "structural_features": {"vertical_strokes": 1}
  }
}
```
For processing images, each stage can be a data URL, a raw base64 string, or an object containing `image`, `url`, `data`, `base64`, `width`, and `height`. If your backend uses different field names, adapt the normalizer in `src/services/recognitionApi.ts`.

### Status handling
The frontend recognizes `success`, `uncertain`, and `invalid` status values. If your backend returns `valid: false`, it is treated as invalid. Otherwise, an available `predicted_digit`/`digit` is displayed as a successful result. No confidence percentage is invented.

## Backend CORS
If frontend and backend run on different ports, enable CORS in FastAPI for `http://localhost:5173` (and your deployed frontend origin).

## Build
```bash
npm run build
npm run preview
```

## Notes
- Upload accepts PNG, JPG and JPEG up to 10 MB.
- No database, authentication, localStorage or cloud storage.
- Optional recent history lives only in React memory and clears on refresh.
