import os, sqlite3, numpy as np
from insightface.app import FaceAnalysis

class AadhaarSystem:
    def __init__(self):
        self.conn = sqlite3.connect("db.db", check_same_thread=False)
        self._create_tables()

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640,640), det_thresh=0.3)  # faster

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
            embedding BLOB
        )
        """)
        self.conn.commit()

    def _load(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT * FROM persons").fetchall()

        self.known_embeddings = []
        self.known_meta = []

        for row in rows:
            aad, name, dob, gender, address, blob = row
            emb = np.frombuffer(blob, dtype=np.float32)

            self.known_embeddings.append(emb)
            self.known_meta.append({
                "aadhaar": aad,
                "name": name,
                "dob": dob,
                "gender": gender,
                "address": address
            })

    def register(self, img, aadhaar, name, dob=None, gender=None, address=None):
        faces = self.app.get(img)
        print("DEBUG: Faces detected:", len(faces))
        if not faces:
            return False, "No face detected"

        emb = faces[0].embedding.astype(np.float32)

        cur = self.conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO persons VALUES(?,?,?,?,?,?)
        """, (aadhaar, name, dob, gender, address, emb.tobytes()))
        self.conn.commit()

        self._load()
        return True, "Registered successfully"

    def recognize(self, img):
        faces = self.app.get(img)
        results = []

        for f in faces:
            emb = f.embedding.astype(np.float32)
            emb /= (np.linalg.norm(emb) + 1e-10)

            best_idx = -1
            best_score = -1

            for i, k in enumerate(self.known_embeddings):
                k = k / (np.linalg.norm(k) + 1e-10)
                sim = float(np.dot(emb, k))

                if sim > best_score:
                    best_score = sim
                    best_idx = i

            if best_score > 0.45:
                meta = self.known_meta[best_idx]
            else:
                meta = {
                    "name": "Unknown",
                    "aadhaar": None,
                    "dob": None,
                    "gender": None,
                    "address": None
                }

            x1, y1, x2, y2 = map(int, f.bbox)

            results.append({
                "name": meta["name"],
                "aadhaar": meta["aadhaar"],
                "dob": meta["dob"],
                "gender": meta["gender"],
                "address": meta["address"],
                "confidence": round(best_score, 3),
                "bbox": [x1, y1, x2, y2]
            })

        return results