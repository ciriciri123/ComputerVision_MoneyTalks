"""
crop_utils.py
-------------
Smart cropping utilities for IDR banknote denomination detection.
Designed to slot into the existing inference.py pipeline.

Two main functions:
  - get_orb_and_color_features()  : drop-in replacement, improved keypoint crop
  - get_denomination_crops()      : returns separate crops optimised for SVM and OCR
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rotate_to_landscape(img: np.ndarray) -> np.ndarray:
    """Rotate portrait image to landscape (banknotes are always wider than tall)."""
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def _keypoint_bbox(img: np.ndarray, orb, clahe,
                   low_pct: float = 8, high_pct: float = 92,
                   pad_ratio: float = 0.18):
    """
    Detect ORB keypoints and return the bounding box that encloses them
    (percentile-trimmed to ignore scattered outlier keypoints).

    Returns (x1, y1, x2, y2) or None if too few keypoints.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    kps, _ = orb.detectAndCompute(blurred, None)

    if not kps or len(kps) < 30:
        return None

    pts = np.array([kp.pt for kp in kps])
    x_min, y_min = np.percentile(pts, low_pct,  axis=0)
    x_max, y_max = np.percentile(pts, high_pct, axis=0)

    pad_x = int((x_max - x_min) * pad_ratio)
    pad_y = int((y_max - y_min) * pad_ratio)

    x1 = max(0, int(x_min) - pad_x)
    y1 = max(0, int(y_min) - pad_y)
    x2 = min(W, int(x_max) + pad_x)
    y2 = min(H, int(y_max) + pad_y)
    return x1, y1, x2, y2


def _contour_bbox(img: np.ndarray, clahe):
    """
    Fallback crop using largest contour (good when ORB finds few keypoints,
    e.g. plain/damaged notes).

    Returns (x1, y1, x2, y2) or None.
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Keep contours that are at least 10% of image area
    min_area = H * W * 0.10
    big = [c for c in contours if cv2.contourArea(c) > min_area]
    if not big:
        big = contours

    all_pts = np.vstack(big)
    x, y, w, h = cv2.boundingRect(all_pts)

    pad = int(min(w, h) * 0.05)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(W, x + w + pad)
    y2 = min(H, y + h + pad)
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Denomination zone crops (the key addition for your use case)
# ---------------------------------------------------------------------------

# IDR banknotes print the denomination number in these relative zones:
#   - bottom-left corner  (primary OCR target)
#   - bottom-right corner (secondary OCR target)
#   - center-left area    (large numeral on most series)
#
# Zones are defined as (x_start_ratio, y_start_ratio, x_end_ratio, y_end_ratio)
# on a landscape-oriented, already-cropped banknote image.

DENOM_ZONES = {
    "bottom_left":  (0.00, 0.55, 0.35, 1.00),   # strongest — largest printed number
    "bottom_right": (0.65, 0.55, 1.00, 1.00),   # mirror corner, 2nd choice
    "center_left":  (0.08, 0.20, 0.50, 0.80),   # large center numeral
    "full":         (0.00, 0.00, 1.00, 1.00),   # fallback
}


def get_zone_crop(bill_img: np.ndarray, zone_key: str) -> np.ndarray:
    """
    Extract a named denomination zone from a landscape bill image.
    zone_key must be one of DENOM_ZONES keys.
    """
    h, w = bill_img.shape[:2]
    x0r, y0r, x1r, y1r = DENOM_ZONES[zone_key]
    x0 = int(x0r * w);  y0 = int(y0r * h)
    x1 = int(x1r * w);  y1 = int(y1r * h)
    return bill_img[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Drop-in replacement for get_orb_and_color_features()
# ---------------------------------------------------------------------------

def get_orb_and_color_features(img: np.ndarray, max_features: int = 2000):
    """
    Improved drop-in replacement for the original function in inference.py.

    Changes vs original:
      - Adds contour-based fallback when ORB finds < 30 keypoints
      - Auto-rotates to landscape before any processing
      - Returns the same 5-tuple: (keypoints, descriptors, color_hist, box_coords, ocr_crop)
    """
    H, W = img.shape[:2]
    orb   = cv2.ORB_create(nfeatures=max_features)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # --- 1. Attempt keypoint-based crop ---
    bbox = _keypoint_bbox(img, orb, clahe)

    # --- 2. Fallback: contour crop ---
    if bbox is None:
        bbox = _contour_bbox(img, clahe)

    box_coords  = None
    money_crop  = img

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        money_crop = img[y1:y2, x1:x2]
        box_coords = [
            float(x1 / W), float(y1 / H),
            float((x2 - x1) / W), float((y2 - y1) / H),
        ]

    # --- 3. Rotate to landscape ---
    money_crop = _rotate_to_landscape(money_crop)
    ocr_crop   = money_crop.copy()

    # --- 4. SVM feature extraction (unchanged from original) ---
    svm_crop     = cv2.resize(money_crop, (800, 400))
    gray_svm     = cv2.cvtColor(svm_crop, cv2.COLOR_BGR2GRAY)
    clahe_svm    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_svm = clahe_svm.apply(gray_svm)
    blurred_svm  = cv2.GaussianBlur(enhanced_svm, (5, 5), 0)

    orb_svm = cv2.ORB_create(nfeatures=max_features)
    final_keypoints, final_descriptors = orb_svm.detectAndCompute(blurred_svm, None)
    if final_descriptors is None:
        final_descriptors = np.array([])

    hsv  = cv2.cvtColor(svm_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    color_hist = hist.flatten()

    return final_keypoints, final_descriptors, color_hist, box_coords, ocr_crop


# ---------------------------------------------------------------------------
# New: multi-zone OCR crop generator
# ---------------------------------------------------------------------------

def get_denomination_crops(img: np.ndarray, max_features: int = 2000):
    """
    Returns a dict of crops, each optimised for a different stage:

        {
            "svm_crop"       : np.ndarray,  # 800x400, for BoVW/SVM (same as original)
            "ocr_full"       : np.ndarray,  # full bill, landscape
            "ocr_bottom_left": np.ndarray,  # bottom-left corner zone
            "ocr_bottom_right":np.ndarray,  # bottom-right corner zone
            "ocr_center_left": np.ndarray,  # large center numeral zone
            "box_coords"     : list | None, # [x, y, w, h] normalised
        }

    Usage in predict_currency():
        crops = get_denomination_crops(img)
        # pass crops["ocr_bottom_left"] first, fallback to crops["ocr_full"]
        # pass crops["svm_crop"] to BoVW pipeline
    """
    orb   = cv2.ORB_create(nfeatures=max_features)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    H, W  = img.shape[:2]

    # --- Localise the bill in the frame ---
    bbox = _keypoint_bbox(img, orb, clahe)
    if bbox is None:
        bbox = _contour_bbox(img, clahe)

    box_coords = None
    bill_img   = img

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bill_img   = img[y1:y2, x1:x2]
        box_coords = [
            float(x1 / W), float(y1 / H),
            float((x2 - x1) / W), float((y2 - y1) / H),
        ]

    bill_img = _rotate_to_landscape(bill_img)

    # --- SVM crop (fixed 800x400, CLAHE + blur already applied in caller) ---
    svm_crop = cv2.resize(bill_img, (800, 400))

    # --- OCR crops: fixed minimum size so Tesseract has enough pixels ---
    MIN_OCR_DIM = 120   # px — below this Tesseract degrades badly

    def _safe_zone(zone_key):
        crop = get_zone_crop(bill_img, zone_key)
        ch, cw = crop.shape[:2]
        if ch < MIN_OCR_DIM or cw < MIN_OCR_DIM:
            scale = max(MIN_OCR_DIM / ch, MIN_OCR_DIM / cw)
            crop  = cv2.resize(crop,
                               (int(cw * scale), int(ch * scale)),
                               interpolation=cv2.INTER_LINEAR)
        return crop

    return {
        "svm_crop":        svm_crop,
        "ocr_full":        bill_img.copy(),
        "ocr_bottom_left": _safe_zone("bottom_left"),
        "ocr_bottom_right":_safe_zone("bottom_right"),
        "ocr_center_left": _safe_zone("center_left"),
        "box_coords":      box_coords,
    }


# ---------------------------------------------------------------------------
# How to wire get_denomination_crops() into predict_currency()
# ---------------------------------------------------------------------------
#
# Replace the block in predict_currency() that reads:
#
#   keypoints, descriptors, color_hist, box_coords, ocr_crop = \
#       get_orb_and_color_features(img)
#
# with:
#
#   from crop_utils import get_denomination_crops
#
#   crops      = get_denomination_crops(img)
#   ocr_crop   = crops["ocr_bottom_left"]   # primary OCR target
#   svm_input  = crops["svm_crop"]
#   box_coords = crops["box_coords"]
#
#   # Extract SVM features from svm_input as before:
#   gray_svm     = cv2.cvtColor(svm_input, cv2.COLOR_BGR2GRAY)
#   clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#   enhanced_svm = clahe.apply(gray_svm)
#   blurred_svm  = cv2.GaussianBlur(enhanced_svm, (5,5), 0)
#   orb          = cv2.ORB_create(nfeatures=2000)
#   final_keypoints, final_descriptors = orb.detectAndCompute(blurred_svm, None)
#   ...
#
# Then in the OCR section, try zones in priority order:
#
#   for zone_key in ("ocr_bottom_left", "ocr_bottom_right",
#                    "ocr_center_left", "ocr_full"):
#       label, ratio, _ = _predict_with_ocr(crops[zone_key])
#       if label and ratio >= 0.30:
#           ocr_label      = label
#           ocr_vote_ratio = ratio
#           break
#
# This way the OCR sees the clearest possible view of the denomination number
# first, before falling back to noisier wider crops.
