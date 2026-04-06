from flask import Flask, request, jsonify, render_template
import base64, cv2, numpy as np
from system import AadhaarSystem
from aadhaar_ocr import extract_aadhaar_details

app = Flask(__name__)
system = AadhaarSystem()

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
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None

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

    ocr = {}
    if aadhaar_img is not None:
        ocr = extract_aadhaar_details(aadhaar_img)

    aadhaar = data.get("aadhaar") or ocr.get("aadhaar_number")
    name = data.get("name") or ocr.get("name")

    print("AADHAAR:", aadhaar)
    print("NAME:", name)

    if face_img is None:
        return jsonify({"ok": False, "msg": "Face image missing"}), 400

    if not aadhaar:
        return jsonify({"ok": False, "msg": "Aadhaar missing"}), 400

    ok, msg = system.register(
        face_img,
        aadhaar,
        name,
        data.get("dob"),
        data.get("gender"),
        data.get("address")
    )

    return jsonify({"ok": ok, "msg": msg, "ocr": ocr})

@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.json
    img = decode(data.get("image"))

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    return jsonify(system.recognize(img))

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=True)