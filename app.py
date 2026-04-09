from flask import Flask, request, jsonify, render_template
import base64, cv2, numpy as np
from system import AadhaarSystem
from aadhaar_ocr import extract_aadhaar_details

app = Flask(__name__)
system = AadhaarSystem()

from threading import Lock
register_lock = Lock() 


def decode(img_b64):
    try:
        if not img_b64:
            return None

        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        img = base64.b64decode(img_b64)
        arr = np.frombuffer(img, np.uint8)

        if arr.size == 0:
            return None

        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if img is None:
            return None

        return np.array(img, copy=True)

    except:
        return None
    

@app.route("/detect_preview", methods=["POST"])
def detect_preview():

    if register_lock.locked():
        return jsonify({"faces": []})

    data = request.get_json()
    img = decode(data["image"])

    if img is None:
        return jsonify({"faces": []})

    try:
        faces = system.app.get(img)
    except:
        return jsonify({"faces": []})

    result = []
    for f in faces:
        x1, y1, x2, y2 = map(int, f.bbox)
        result.append({"bbox": [x1, y1, x2, y2]})

    return jsonify({"faces": result})


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/register")
def register_page():
    return render_template("register.html")


@app.route("/recognize")
def recognize_page():
    return render_template("recognize.html")


@app.route("/api/register", methods=["POST"])
def register():
    data = request.json

    face_img = decode(data.get("image"))
    aadhaar_img = decode(data.get("aadhaar_photo"))

    print("FACE IMG:", face_img is not None)
    print("AADHAAR IMG:", aadhaar_img is not None)

    if face_img is None:
        return jsonify({"ok": False, "msg": "Face image missing"}), 400

    # KEEP ORIGINAL (NO DISTORTION)
    orig_face = face_img.copy()
    orig_aadhaar = aadhaar_img.copy() if aadhaar_img is not None else None

    # ONLY FOR DETECTION (NOT STORAGE)
    face_small = cv2.resize(face_img, (320, 320))

    # disable OCR (for stability)
    ocr = {}

    aadhaar = data.get("aadhaar") or ocr.get("aadhaar_number")
    name = data.get("name") or ocr.get("name")

    if not aadhaar:
        return jsonify({"ok": False, "msg": "Aadhaar missing"}), 400

    ok, msg = system.register(
        orig_face,          # ✅ ORIGINAL IMAGE (FIXED)
        aadhaar,
        name,
        data.get("dob"),
        data.get("gender"),
        data.get("address"),
        orig_aadhaar       # ✅ ORIGINAL AADHAAR IMAGE (FIXED)
    )

    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.json
    img = decode(data.get("image"))

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    return jsonify(system.recognize(img))


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)