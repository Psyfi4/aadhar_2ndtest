# aadhaar_ocr.py (add or replace)
import re
import cv2
import numpy as np
import pytesseract
from PIL import Image

# helper: clean text
def _clean_text(s: str) -> str:
    if s is None:
        return ''
    # remove weird chars but keep alphanum and some punctuation
    s = s.replace('\xa0', ' ')
    s = re.sub(r'[^\x00-\x7F]+', ' ', s)  # remove non-ascii noise
    s = re.sub(r'[^\w\s\-/,:.]', ' ', s)  # keep useful chars
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _find_aadhaar_number(text: str):
    # Aadhaar is 12 digits, often grouped in 4-4-4. Find continuous 12 digits or groups.
    if not text:
        return None
    # remove spaces and non-digits and search for 12 digits contiguous
    digits = re.findall(r'\d', text)
    if len(digits) >= 12:
        # try to find contiguous 12-digit block
        m = re.search(r'(\d{4}\s?\d{4}\s?\d{4})', text)
        if m:
            return re.sub(r'\s+', '', m.group(1))
        # fallback: first 12 digits in the text
        return ''.join(digits[:12])
    return None

def _find_date(text: str):
    # common date formats dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy
    m = re.search(r'(\d{1,2}[\/\-.]\d{1,2}[\/\-.](?:\d{2,4}))', text)
    return m.group(1) if m else None

def _find_name(text: str):
    # Heuristic: Aadhaar name often in lines with letters and spaces, uppercase words,
    # try to find a line with >=2 capitalized words and no digits.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidate = None
    for line in lines:
        if any(ch.isdigit() for ch in line):
            continue
        words = [w for w in re.split(r'[^A-Za-z]+', line) if w]
        if len(words) >= 2:
            # pick the line with the most alphabetic chars
            if candidate is None or len(line) > len(candidate):
                candidate = line
    if candidate:
        # clean and return
        candidate = re.sub(r'[^A-Za-z\s]', ' ', candidate)
        candidate = re.sub(r'\s+', ' ', candidate).strip()
        # ensure plausible length
        if 2 <= len(candidate.split()) <= 6:
            return candidate.title()
    return None

def scan_aadhaar_image_from_pil(pil_image: Image.Image) -> dict:
    """
    Input: PIL Image (RGB)
    Returns: dict with keys: aadhaar_number, name, date_of_birth, raw_text, gender, address (best-effort)
    """
    # convert to OpenCV BGR
    img = np.array(pil_image.convert('RGB'))[:, :, ::-1].copy()

    # 1) Resize to reasonable width for OCR if very large
    h, w = img.shape[:2]
    if w > 2000:
        scale = 2000.0 / w
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # 2) convert to gray and denoise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 3) contrast/brightness normalize via CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # 4) adaptive threshold to enhance text
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)

    # 5) morphological opening to reduce small noise, then closing to fill letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    proc = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    proc = cv2.morphologyEx(proc, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 6) try OCR on the processed image
    custom_config = r'--oem 3 --psm 6'  # Default LSTM, assume a uniform block of text
    try:
        raw = pytesseract.image_to_string(proc, config=custom_config)
    except Exception:
        # fallback to plain grayscale PIL->pytesseract
        raw = pytesseract.image_to_string(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                                          config=custom_config)

    raw_clean = _clean_text(raw)

    aadhaar = _find_aadhaar_number(raw_clean)
    dob = _find_date(raw_clean)
    name = _find_name(raw_clean)

    # Try some fallback passes: a) directly OCR on gray, b) on original color (if above failed)
    if not aadhaar or not name:
        try:
            raw2 = pytesseract.image_to_string(gray, config=custom_config)
            raw2 = _clean_text(raw2)
            if not aadhaar:
                aadhaar = _find_aadhaar_number(raw2)
            if not name:
                name = name or _find_name(raw2)
            if not dob:
                dob = dob or _find_date(raw2)
            raw_clean += "\n" + raw2
        except Exception:
            pass

    # final normalization: date to dd/mm/yyyy if possible
    if dob:
        # normalize separators
        dob = re.sub(r'[\-\.]', '/', dob)
        # if year is 2-digit, try to make 4-digit (assume 19xx/20xx fallback) — leave as-is for now
        dob = dob

    result = {
        'aadhaar_number': aadhaar,
        'name': name,
        'date_of_birth': dob,
        'gender': None,
        'address': None,
        'raw_text': raw_clean
    }
    return result

