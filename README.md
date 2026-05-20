# MoneyTalks — Web-based Currency Detector for the Visually Impaired

MoneyTalks is a browser-first assistive technology application that enables visually impaired users to independently identify Indonesian Rupiah banknote denominations in real time. The device camera streams to a Flask backend, where a classical computer vision pipeline (ORB → BoVW → SVM + Tesseract OCR) classifies the denomination and returns an audio announcement via the Web Speech API.

**Team:**
- Antonio Darren Novianto Saputra — 2802554634
- Dominicius Francis Ang Gunadi — 2802561293
- Frans Sebastian Winata — 2802489414
- Garren Tanavaro — 2802516182
- Riccy Riandy Intan — 2802547856

---

## Features

- Real-time banknote detection via rear camera (MediaDevices API)
- SVM classifier fusing ORB texture features (BoVW + TF-IDF) with HSV colour histograms
- Tesseract OCR secondary prediction with confidence-based decision fusion
- Audio output via Web Speech API (gTTS server-side fallback)
- Scan persistence to Supabase (image storage + PostgreSQL metadata)
- Admin panel: scan browser, model version management with zero-downtime hot-swap

---

## Project Structure

```
MoneyTalks/
├── app.py                  # Flask entry point + all routes
├── inference.py            # SVM + OCR inference pipeline
├── supabase_client.py      # Supabase DB & Storage operations
├── requirements.txt
├── .env.example            # Environment variable template
│
├── src/
│   ├── baseline/           # Baseline training pipeline
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   └── model.py
│   └── proposed/           # Proposed training pipeline
│       ├── preprocessing.py
│       ├── features.py
│       └── model.py
│
├── scripts/
│   ├── augment.py          # Balanced data augmentation
│   └── clean_augment.py    # Remove augmented files
│
├── models/
│   ├── baseline/           # bovw_dictionary.pkl, svm_model.pkl
│   └── proposed/           # bovw_dictionary.pkl, svm_model.pkl, tfidf_scaler.pkl
│
├── templates/
│   ├── index.html          # Main camera detection UI
│   ├── test.html           # Static image upload test page
│   └── admin/              # Admin panel templates
│
├── tests/                  # pytest unit tests (89 tests)
│   ├── conftest.py
│   ├── test_app.py
│   ├── test_inference.py
│   ├── test_preprocessing.py
│   └── test_features.py
│
├── migrations/
│   └── 001_initial_schema.sql  # Supabase DB schema
│
└── docs/
    ├── PRD.md
    ├── MoneyTalks_PRD_v2.docx
    └── Money Talks_ Computer Vision Project.docx
```

---

## Methodology

### Baseline Model
- **Preprocessing:** Resize → Grayscale → Gaussian Blur
- **Features:** ORB descriptors → BoVW (KMeans, 800 clusters)
- **Classifier:** SVM with linear kernel

### Proposed Model
- **Preprocessing:** Resize (800×400) → BGR colour retained
- **Features:** ORB + CLAHE → BoVW (800 clusters) with TF-IDF weighting (×3.0) fused with HSV colour histogram (8×8×8)
- **Classifier:** SVM with linear kernel, balanced class weights
- **Inference:** Smart bounding box crop → SVM prediction → Tesseract OCR on 2 ROIs → confidence-based decision fusion

### Data Augmentation (`scripts/augment.py`)
Balanced oversampling via: dark (α=0.75), bright (α=1.1), Gaussian blur (5×5), random noise, and 180° rotation.

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

> **Windows:** Tesseract must be installed separately from [github.com/UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki). Update the path in `inference.py` line 11 if needed.

### 2. Configure environment variables
```bash
cp .env.example .env
```
Fill in `.env` with your Supabase credentials (see `.env.example` for details).

### 3. Set up Supabase
1. Run `migrations/001_initial_schema.sql` in the Supabase SQL editor
2. Create two private storage buckets: `scanned-images` and `model-files`
3. Seed an admin account:
```bash
python -c "import bcrypt; print(bcrypt.hashpw(b'yourpassword', bcrypt.gensalt(12)).decode())"
```
```sql
INSERT INTO "Administrator" (email, password_hash) VALUES ('you@example.com', '<hash>');
```

### 4. Train the models (optional — pre-trained models included)
Place your dataset in `data/` with subfolders per class (`idr_1000`, `idr_2000`, ..., `idr_100000`).

```bash
# Proposed model (recommended)
python src/proposed/model.py

# Baseline model
python src/baseline/model.py

# Data augmentation
python scripts/augment.py
```

---

## Running the Application

```bash
python app.py
```

| URL | Description |
|---|---|
| `http://localhost:5000` | Main camera detection app |
| `http://localhost:5000/test` | Static image upload test |
| `http://localhost:5000/admin/login` | Admin panel |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

89 tests covering: OCR text normalisation, Levenshtein distance, denomination extraction, image preprocessing, BoVW histogram generation, ORB feature extraction, and all Flask API endpoints.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/detect` | Classify a banknote frame |
| `POST` | `/api/upload-image` | Persist a scan to Supabase |
| `GET` | `/api/tts?text=...` | gTTS audio fallback |

### `POST /api/detect` response
```json
{
  "valid": true,
  "message": "Lima Puluh Ribu Rupiah",
  "raw_label": "idr_50000",
  "confidence": 0.923,
  "box": [0.12, 0.08, 0.76, 0.84]
}
```

---

## Dataset

[Rupiah Banknotes Dataset](https://github.com/mkaspulanwar/rupiah-banknotes-dataset) by mkaspulanwar — MIT License.  
Classes: IDR 1,000 / 2,000 / 5,000 / 10,000 / 20,000 / 50,000 / 100,000.
