import easyocr, re, cv2

reader = easyocr.Reader(['en'], gpu=False)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return th

def extract_aadhaar_details(img):
    img = preprocess(img)
    results = reader.readtext(img)

    text = " ".join([r[1] for r in results])

    aadhaar = None
    m = re.search(r"\d{4}\s?\d{4}\s?\d{4}", text)
    if m:
        aadhaar = m.group(0).replace(" ", "")

    dob = None
    m = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if m:
        dob = m.group(0)

    gender = None
    if "MALE" in text.upper():
        gender = "MALE"
    elif "FEMALE" in text.upper():
        gender = "FEMALE"

    name = None
    words = text.split()
    for w in words:
        if w.isalpha() and len(w) > 3:
            name = w
            break

    print("OCR TEXT:", text)

    return {
        "aadhaar_number": aadhaar,
        "name": name,
        "dob": dob,
        "gender": gender,
        "raw_text": text
    }