import unicodedata
import cv2
import numpy as np

def vn_no_diacritics(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    nfkd = unicodedata.normalize("NFD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def preprocess_face(img):
    img = cv2.resize(img, (200, 200))
    img = cv2.equalizeHist(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.bilateralFilter(img, 5, 50, 50)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img

# Bộ phát hiện khuôn mặt Haar Cascade dùng chung
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)