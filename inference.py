import os
import io
import cv2
import numpy as np
import joblib
import pytesseract
from PIL import Image

# MENGHUBUNGKAN PYTHON KE TESSERACT
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'proposed')

_models_loaded = False
_kmeans_model = None
_tfidf = None
_svm = None

def load_models():
    global _models_loaded, _kmeans_model, _tfidf, _svm
    if _models_loaded: return
    bovw_path = os.path.join(MODELS_DIR, 'bovw_dictionary.pkl')
    tfidf_path = os.path.join(MODELS_DIR, 'tfidf_scaler.pkl')
    svm_path = os.path.join(MODELS_DIR, 'svm_model.pkl')
    
    _kmeans_model = joblib.load(bovw_path)
    _tfidf = joblib.load(tfidf_path)
    _svm = joblib.load(svm_path)
    _models_loaded = True

def preprocess_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    cv_img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if cv_img is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img

def get_orb_and_color_features(img, max_features=2000):
    # --- 1. MAGIC FIX: CANNY EDGE DENGAN HEAVY BLUR (ANTI-MEJA) ---
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Blur sangat tebal agar motif meja marmer menghilang, sisa bentuk uang saja
    blurred_full = cv2.GaussianBlur(gray_full, (15, 15), 0)
    edges = cv2.Canny(blurred_full, 30, 100)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    H, W = img.shape[:2]
    box_coords = None
    money_crop = img
    
    if contours:
        # Ambil garis tepi paling besar di layar
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 15000:
            x, y, w, h = cv2.boundingRect(largest_contour)
            money_crop = img[y:y+h, x:x+w]
            box_coords = [float(x/W), float(y/H), float(w/W), float(h/H)]
            
    # --- 2. AUTO-ROTATE ---
    if money_crop.shape[0] > money_crop.shape[1]:
        money_crop = cv2.rotate(money_crop, cv2.ROTATE_90_CLOCKWISE)
        
    # --- 3. SIMPAN POTONGAN ASLI UNTUK OCR (ANTI-GEPENG!) ---
    ocr_crop = money_crop.copy()
        
    # --- 4. RESIZE KHUSUS UNTUK SVM ---
    svm_crop = cv2.resize(money_crop, (800, 400))
    
    orb = cv2.ORB_create(nfeatures=max_features)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_svm = cv2.cvtColor(svm_crop, cv2.COLOR_BGR2GRAY)
    enhanced_svm = clahe.apply(gray_svm)
    blurred_svm = cv2.GaussianBlur(enhanced_svm, (5, 5), 0)
    
    final_keypoints, final_descriptors = orb.detectAndCompute(blurred_svm, None)
    if final_descriptors is None: final_descriptors = np.array([])
        
    hsv = cv2.cvtColor(svm_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist) 
    color_hist = hist.flatten()
        
    # Kita berikan 'ocr_crop' ke luar fungsi
    return final_keypoints, final_descriptors, color_hist, box_coords, ocr_crop

def get_bovw_histogram(descriptors, kmeans_model):
    num_clusters = kmeans_model.n_clusters
    if len(descriptors) > 0:
        words = kmeans_model.predict(descriptors)
        hist, _ = np.histogram(words, bins=np.arange(num_clusters + 1))
    else:
        hist = np.zeros(num_clusters)
    return hist

def predict_currency(image_bytes):
    if not _models_loaded: load_models()

    try:
        img = preprocess_image(image_bytes)
        keypoints, descriptors, color_hist, box_coords, ocr_crop = get_orb_and_color_features(img)

        if len(descriptors) < 50:
            return {"label": "none", "confidence": 0.0}

        # --- 1. PREDIKSI SVM ---
        bovw_raw = get_bovw_histogram(descriptors, _kmeans_model)
        bovw_tfidf = _tfidf.transform([bovw_raw]).toarray()[0]
        bovw_tfidf = bovw_tfidf * 3.0
        
        fused = np.hstack((bovw_tfidf, color_hist))
        raw_label = _svm.predict([fused])[0]
        svm_label = str(raw_label)
        
        proba = _svm.predict_proba([fused])[0]
        confidence = np.max(proba)

        # --- 2. PREDIKSI OCR ---
        ocr_label = None
        try:
            # MAGIC FIX 2: ADAPTIVE THRESHOLD (ANTI SILAU)
            H_ocr, W_ocr = ocr_crop.shape[:2]
            # Zoom proporsional 2x lipat
            zoom_ocr = cv2.resize(ocr_crop, (W_ocr*2, H_ocr*2), interpolation=cv2.INTER_CUBIC)
            
            gray_ocr = cv2.cvtColor(zoom_ocr, cv2.COLOR_BGR2GRAY)
            blur_ocr = cv2.GaussianBlur(gray_ocr, (5, 5), 0)
            
            # Adaptive Threshold sangat bagus untuk cahaya tidak rata (flash kamera)
            thresh_ocr = cv2.adaptiveThreshold(blur_ocr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
            
            ocr_text = pytesseract.image_to_string(thresh_ocr, config='--psm 11 -c tessedit_char_whitelist=0123456789')
            cleaned_text = ocr_text.strip().replace('\n', ' ')
            print(f"[OCR RAW TEXT] Hasil pembacaan: {cleaned_text}")
            
            denominations = {'100000': 'idr_100000', '50000': 'idr_50000', '20000': 'idr_20000', 
                             '10000': 'idr_10000', '5000': 'idr_5000', '2000': 'idr_2000', '1000': 'idr_1000'}
            
            # Kita pisahkan jadi kata per kata agar 2000 tidak terbaca saat sebenarnya 20000
            words = cleaned_text.split()
            for denom, label in denominations.items():
                if denom in words or denom in cleaned_text:
                    # Validasi Ekstra: Cegah salah paham angka nol
                    if denom == '2000' and '20000' in cleaned_text: continue
                    if denom == '1000' and '10000' in cleaned_text: continue
                    if denom == '1000' and '100000' in cleaned_text: continue
                    if denom == '10000' and '100000' in cleaned_text: continue
                    
                    ocr_label = label
                    print(f"[OCR SUCCESS] Angka {denom} ditemukan!")
                    break
        except Exception as e:
            print(f"[OCR WARNING] Tesseract error: {e}")

        # --- 3. DECISION FUSION ---
        final_label = svm_label
        
        # Hakim OCR Menyelamatkan Hari!
        if ocr_label and (confidence < 0.90 or ocr_label != svm_label):
            final_label = ocr_label
            confidence = 0.99 
            print(f"[FUSION] OCR Overwrite! Mengubah prediksi menjadi {ocr_label}")

        result = {
            'label': final_label,
            'confidence': float(confidence)
        }
        if box_coords: result['box'] = box_coords
            
        return result
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return {"error": str(e)}