from flask import Flask, request, jsonify, render_template
import base64, cv2, numpy as np
from system import AadhaarSystem
from aadhaar_ocr import extract_aadhaar_details

app = Flask(__name__)
system = AadhaarSystem()

def decode(img_b64):
    img = base64.b64decode(img_b64.split(",")[1])
    arr = np.frombuffer(img, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/recognize")
def recognize_page():
    return render_template("recognize.html")

@app.route("/detect_preview", methods=["POST"])
def detect_preview():
    data = request.json
    img = decode(data["image"])

    faces = system.app.get(img)

    results = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        results.append({
            "bbox": [x1, y1, x2, y2],
            "w": x2 - x1,
            "h": y2 - y1
        })

    return jsonify({"faces": results})

# -------- REGISTER API --------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    img = decode(data["image"])

    def preprocess(img):
        import cv2
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return th

# OCR extraction
    ocr_data = extract_aadhaar_details(img)
    aadhaar = ocr_data["aadhaar_number"]

    aadhaar = data.get("aadhaar") or ocr_data["aadhaar_number"]
    name = data.get("name") or ocr_data["name"]

    ok, msg = system.register(img, aadhaar, name)

    ocr_data = extract_aadhaar_details(preprocess(img))

    return jsonify({
        "ok": ok,
        "msg": msg,
        "ocr": ocr_data
})

# -------- RECOGNIZE API --------
@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.json
    img = decode(data["image"])
    res = system.recognize(img)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
