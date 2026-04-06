import easyocr
import re

reader = easyocr.Reader(['en'], gpu=False)

def extract_aadhaar(img):
    results = reader.readtext(img)
    text = " ".join([r[1] for r in results])

    match = re.search(r"\d{4}\s?\d{4}\s?\d{4}", text)

    if match:
        return match.group(0).replace(" ", "")
    return None
