# ComputerVision_MoneyTalks

MoneyTalks is a Computer Vision project designed to recognize and classify currency/money from images. It compares a Baseline approach with a Proposed enhanced approach using Bag of Visual Words (BoVW), feature fusion, and Support Vector Machines (SVM). 

The application also includes a web interface component using Streamlit and a backend service structure.

## Project Structure

```
ComputerVision_MoneyTalks/
├── app/                  # Web application backend and services
│   ├── auth.py           # Authentication logic
│   ├── routes.py         # API or app routes
│   └── services/         # Core services including model inference and Supabase client
│       ├── inference.py
│       └── supabase_client.py
├── models/               # Saved trained models and preprocessors (.pkl)
│   ├── baseline/         # Baseline model artifacts (BoVW dictionary, SVM model)
│   └── proposed/         # Proposed model artifacts (BoVW dictionary, Scaler, TF-IDF, SVM)
├── src/                  # Source code for training and evaluation
│   ├── baseline/         # Baseline methodology scripts
│   │   ├── features.py
│   │   ├── model.py
│   │   └── preprocessing.py
│   └── proposed/         # Proposed improved methodology scripts
│       ├── features.py
│       ├── model.py
│       └── preprocessing.py
├── data/                 # Dataset directory (create this and add your data here)
├── app.py                # Main application entry point
├── requirements.txt      # Project dependencies
└── README.md             # This file

```

## Methodology

### 1. Baseline Model
The baseline approach uses traditional computer vision techniques to classify images:
*   **Feature Extraction:** ORB (Oriented FAST and Rotated BRIEF) descriptors.
*   **Representation:** Bag of Visual Words (BoVW) utilizing KMeans clustering (150 clusters).
*   **Classification:** Multi-class Support Vector Machine (SVM) with RBF kernel and a K-Nearest Neighbors (KNN) comparison.

### 2. Proposed Model
The proposed model enhances the baseline with feature fusion and advanced weighting to improve accuracy:
*   **Feature Extraction:** Fuses ORB descriptors with Color features.
*   **Representation:** Bag of Visual Words (BoVW) (800 clusters).
*   **Weighting & Scaling:** Applies TF-IDF (Term Frequency-Inverse Document Frequency) weighting to the BoVW histograms and normalizes the fused features using `StandardScaler`.
*   **Classification:** Tuned Multi-class SVM with RBF kernel and balanced class weights.

## Requirements

The project uses the following key libraries (see `requirements.txt` for details):
*   `opencv-python`
*   `scikit-learn`
*   `numpy`
*   `pytesseract`
*   `streamlit` & `streamlit-webrtc`
*   `pytest`

### Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment (optional but recommended).
3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Models

Ensure you have your dataset placed in a `data/` directory at the root level before training. The dataset should be structured in subfolders per class.

**To train the Baseline Model:**
```bash
python src/baseline/model.py
```

**To train the Proposed Model:**
```bash
python src/proposed/model.py
```

The trained models and dictionaries will be automatically saved in their respective folders under the `models/` directory.

### Running the Application

*(Instructions for running the web application will go here once `app.py` is fully implemented)*
```bash
streamlit run app.py
```
*(or depending on backend setup, `python app.py`)*

hehehe halooo guys