import os
import io
import re
from collections import Counter
import cv2
import numpy as np
import joblib
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models', 'proposed')

_models_loaded = False
_kmeans_model = None
_tfidf = None
_svm = None

DENOMINATION_MAP = {
    '100000': 'idr_100000',
    '50000': 'idr_50000',
    '20000': 'idr_20000',
    '10000': 'idr_10000',
    '5000': 'idr_5000',
    '2000': 'idr_2000',
    '1000': 'idr_1000'
}
