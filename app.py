from flask import Flask, request, jsonify, render_template
import base64, cv2, numpy as np
from system import AadhaarSystem
from ocr import extract_aadhaar

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

# -------- REGISTER API --------
@app.route("/api/register", methods=["POST"])
def register():
    data = request.json
    img = decode(data["image"])
    aadhaar = data.get("aadhaar")
    if not aadhaar:
    aadhaar = extract_aadhaar(img)
    name = data["name"]

    ok,msg = system.register(img,aadhaar,name)
    return jsonify({"ok":ok,"msg":msg})

# -------- RECOGNIZE API --------
@app.route("/api/recognize", methods=["POST"])
def recognize():
    data = request.json
    img = decode(data["image"])
    res = system.recognize(img)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
