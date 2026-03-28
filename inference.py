import os
import io
import cv2
import numpy as np
import joblib
from PIL import Image

# Path to the models folder
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'proposed')

# Globals to act as a thread-safe singleton
_models_loaded = False
_kmeans_model = None
_tfidf = None
_scaler = None
_svm = None

def load_models():
    global _models_loaded, _kmeans_model, _tfidf, _scaler, _svm
    if _models_loaded:
        return

    print("[*] Loading inference models from disk...")
    # Dictionary path
    bovw_path = os.path.join(MODELS_DIR, 'bovw_dictionary.pkl')
    # TFIDF path
    tfidf_path = os.path.join(MODELS_DIR, 'tfidf_scaler.pkl')
    # Scaler path
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    # SVM path
    svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')

    if not all(os.path.exists(p) for p in [bovw_path, tfidf_path, scaler_path, svm_path]):
        raise FileNotFoundError("One or more .pkl model files are missing in proposed/")

    _kmeans_model = joblib.load(bovw_path)
    _tfidf = joblib.load(tfidf_path)
    _scaler = joblib.load(scaler_path)
    _svm = joblib.load(svm_path)
    _models_loaded = True
    print("[*] Models loaded successfully.")

def preprocess_image(image_bytes, target_size=(800, 400)):
    # Decode directly using OpenCV to perfectly match cv2.imread used in training
    np_img = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # In case imdecode fails, fallback
    if cv_img is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
    return cv2.resize(cv_img, target_size)

def get_orb_and_color_features(img, max_features=2000):
    orb = cv2.ORB_create(nfeatures=max_features)
    
    # 1. Color features (HSV Histogram)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_hist = hist.flatten()
    
    # 2. Texture features (ORB + CLAHE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    keypoints, descriptors = orb.detectAndCompute(blurred, None)
    if descriptors is None:
        descriptors = np.array([])
        
    return descriptors, color_hist

def get_bovw_histogram(descriptors, kmeans_model):
    num_clusters = kmeans_model.n_clusters
    if len(descriptors) > 0:
        words = kmeans_model.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
    else:
        hist = np.zeros(num_clusters)
    return hist

def predict_currency(image_bytes):
    if not _models_loaded:
        load_models()

    try:
        # 1. Preprocess
        img = preprocess_image(image_bytes)
        
        # 2. Extract Features
        descriptors, color_hist = get_orb_and_color_features(img)

        # Reject if not enough features (background/blank/blur)
        if len(descriptors) < 50:
            return {"label": "none", "confidence": 0.0}

        # 3. BoVW Histogram
        bovw_raw = get_bovw_histogram(descriptors, _kmeans_model)
        
        # 4. TF-IDF Scaling
        bovw_tfidf = _tfidf.transform([bovw_raw]).toarray()[0]
        
        # 5. Fuse & Normalize
        fused = np.hstack((bovw_tfidf, color_hist))
        final_features = _scaler.transform([fused])
        
        # 6. Predict
        raw_label = _svm.predict(final_features)[0]
        label = str(raw_label)
        
        # Determine confidence if probability=True was trained, otherwise use decision_function
        confidence = 1.0
        try:
            proba = _svm.predict_proba(final_features)
            confidence = np.max(proba)
        except AttributeError:
            # Fallback to decision_function to approximate confidence
            dec = _svm.decision_function(final_features)
            if len(dec.shape) > 1:
                # multiclass - apply temperature to sharpen probability distribution
                # Adjusted to 0.8 so that "OOD" / Background (which defaults to idr_1000) 
                # scores below exactly 0.75 (approx 0.72), while real matches score above 0.75.
                temperature = 0.35
                scaled_dec = dec / temperature
                scores = np.exp(scaled_dec) / np.sum(np.exp(scaled_dec), axis=1, keepdims=True)
                confidence = np.max(scores)
            else:
                score = 1 / (1 + np.exp(-dec[0]))
                confidence = score if label == _svm.classes_[1] else 1 - score

        return {
            'label': label,
            'confidence': float(confidence)
        }
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return {"error": str(e)}
