import easyocr
import re
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

# ---------- HELPER ----------
def center(box):
    x = (box[0][0] + box[2][0]) / 2
    y = (box[0][1] + box[2][1]) / 2
    return np.array([x, y])

def distance(b1, b2):
    return np.linalg.norm(center(b1) - center(b2))


# ---------- MAIN ----------
def extract_aadhaar_details(img):
    results = reader.readtext(img)

    texts = []
    for bbox, text, conf in results:
        text = text.strip()
        if conf > 0.4:   # filter noise
            texts.append({
                "bbox": bbox,
                "text": text,
                "conf": conf
            })

    full_text = " ".join([t["text"] for t in texts])

    # ---------- Aadhaar ----------
    aadhaar = None
    aadhaar_match = re.search(r"\d{4}\s?\d{4}\s?\d{4}", full_text)
    if aadhaar_match:
        aadhaar = aadhaar_match.group(0).replace(" ", "")

    # ---------- DOB (NEAR 'DOB') ----------
    dob = None
    dob_keywords = ["DOB", "Birth", "Year"]

    for t in texts:
        if any(k.lower() in t["text"].lower() for k in dob_keywords):
            # find closest number-like text
            nearest = None
            min_dist = 9999

            for t2 in texts:
                if re.search(r"\d{2}/\d{2}/\d{4}", t2["text"]):
                    d = distance(t["bbox"], t2["bbox"])
                    if d < min_dist:
                        min_dist = d
                        nearest = t2["text"]

            if nearest:
                dob = nearest
                break

    # fallback regex
    if not dob:
        m = re.search(r"\d{2}/\d{2}/\d{4}", full_text)
        if m:
            dob = m.group(0)

    # ---------- GENDER ----------
    gender = None
    for t in texts:
        if re.search(r"\bMALE\b", t["text"], re.I):
            gender = "MALE"
        elif re.search(r"\bFEMALE\b", t["text"], re.I):
            gender = "FEMALE"

    # ---------- NAME (TOP REGION LOGIC) ----------
    name = None

    # sort by vertical position (top first)
    texts_sorted = sorted(texts, key=lambda x: x["bbox"][0][1])

    blacklist = ["GOVERNMENT", "INDIA", "DOB", "MALE", "FEMALE", "YEAR"]

    for t in texts_sorted:
        word = t["text"].upper()

        # heuristic: names are uppercase & medium length
        if (
            word.isalpha()
            and 3 < len(word) < 20
            and word not in blacklist
        ):
            name = word
            break

    return {
        "aadhaar_number": aadhaar,
        "name": name,
        "dob": dob,
        "gender": gender,
        "raw_text": full_text
    }
