import cv2, sqlite3, numpy as np
import base64
from insightface.app import FaceAnalysis

class AadhaarSystem:
    def __init__(self):
        self.conn = sqlite3.connect("db.db", check_same_thread=False)
        self._create_tables()

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.3)

        self.known_embeddings = []
        self.known_meta = []
        self._load()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS persons(
            aadhaar TEXT PRIMARY KEY,
            name TEXT,
            dob TEXT,
            gender TEXT,
            address TEXT,
            embedding BLOB,
            photo BLOB,
            aadhaar_photo BLOB      
        )
        """)
        self.conn.commit()

    def _load(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT * FROM persons").fetchall()

        self.known_embeddings = []
        self.known_meta = []

        for row in rows:
            aad, name, dob, gender, address, blob, photo, aadhaar_photo = row

            emb = np.frombuffer(blob, dtype=np.float32).copy()  # FIX
            emb /= (np.linalg.norm(emb) + 1e-10)

            self.known_embeddings.append(emb)
            self.known_meta.append({
                "aadhaar": aad,
                "name": name,
                "dob": dob,
                "gender": gender,
                "address": address,
                "photo": photo,
                "aadhaar_photo": aadhaar_photo
            })

    # FIXED REGISTER
    def register(self, face_img, aadhaar, name, dob, gender, address, aadhaar_img):
        try:
            print("REGISTER STARTED")

            face_img = np.array(face_img, copy=True)   # ADD THIS

            faces = self.app.get(face_img)
            print("Faces found:", len(faces))

            if not faces:
                return False, "No face detected"

            emb = faces[0].embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-10)   # ADD THIS

            success, buffer = cv2.imencode('.jpg', face_img)
            if not success:
                return False, "Photo encoding failed"
            photo_blob = buffer.tobytes()

            if aadhaar_img is not None:
                success, buffer = cv2.imencode('.jpg', aadhaar_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if not success:
                    return False, "Aadhaar photo encoding failed"
                aadhaar_blob = buffer.tobytes()
            else:
                aadhaar_blob = None

            cursor = self.conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO persons
            (aadhaar, name, dob, gender, address, embedding, photo, aadhaar_photo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                aadhaar, name, dob, gender, address, emb.tobytes(), photo_blob, aadhaar_blob
            ))

            self.conn.commit()

            self._load()   # IMPORTANT (refresh memory)

            return True, "Registered successfully"

        except Exception as e:
            print("REGISTER ERROR:", str(e))
            return False, str(e)

    # FIXED RECOGNIZE
    def recognize(self, img):
        h, w = img.shape[:2]

    # resize ONLY for detection
        img_small = cv2.resize(img, (320, 240))

        faces = self.app.get(img_small)
        if not faces:
            return []

    # pick largest face
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        f = faces[0]

    # FILTER (RELAXED)
        if hasattr(f, "det_score") and f.det_score < 0.4:
            return []

        x1, y1, x2, y2 = map(int, f.bbox)

    # SCALE BACK TO ORIGINAL SIZE (CRITICAL FIX)
        scale_x = w / 320
        scale_y = h / 240

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        if (x2 - x1) < 60 or (y2 - y1) < 60:
            return []

        emb = f.embedding.astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-10)

        best_idx = -1
        best_score = -1

        for i, k in enumerate(self.known_embeddings):
            sim = float(np.dot(emb, k))
            if sim > best_score:
                best_score = sim
                best_idx = i

        print("MATCH SCORE:", best_score)

        if best_score > 0.6:
            meta = self.known_meta[best_idx]
        else:
            meta = {
            "name": "Unknown",
            "aadhaar": None,
            "dob": None,
            "gender": None,
            "address": None,
            "photo": None,
            "aadhaar_photo": None
        }

        return [{
        "name": meta["name"],
        "aadhaar": meta["aadhaar"],
        "dob": meta["dob"],
        "gender": meta["gender"],
        "address": meta["address"],
        "confidence": round(best_score, 3),
        "bbox": [x1, y1, x2, y2],   # NOW CORRECT
        "photo": base64.b64encode(meta["photo"]).decode() if meta.get("photo") else None,
        "aadhaar_photo": base64.b64encode(meta["aadhaar_photo"]).decode() if meta.get("aadhaar_photo") else None
    }]