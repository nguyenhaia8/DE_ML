# CV Personality Frontend

Vite + React frontend for the CV personality prediction model.

## Run

```bash
npm install
npm run dev
```

## Backend

```bash
cd ../..
python3.13 -m venv app/.venv
app/.venv/bin/python -m pip install -r app/backend/requirements.txt
cd app
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The app expects these backend routes:

- `POST /api/v1/predict` with JSON `{ "text": "...", "model_id": "baseline" }`
- `POST /api/v1/extract/pdf` with multipart field `file`
- `POST /api/v1/predict/pdf` with multipart fields `file` and `model_id`
