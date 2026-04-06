import os
import sqlite3
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

class AadhaarSystem:
    def __init__(self):
        os.makedirs("static/registered_faces", exist_ok=True)
        os.makedirs("static/aadhaar_photos", exist_ok=True)

        self.conn = sqlite3.connect("db.db", check_same_thread=False)
        self._create_tables()

        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=(640,640))

        self.known_embeddings = []
        self.known_meta = []
        self._load()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS persons(
            aadhaar TEXT PRIMARY KEY,
            name TEXT,
            embedding BLOB,
            face_path TEXT,
            aadhaar_path TEXT
        )
        """)
        self.conn.commit()

    def _load(self):
        cur = self.conn.cursor()
        rows = cur.execute("SELECT aadhaar,name,embedding FROM persons").fetchall()
        self.known_embeddings = []
        self.known_meta = []
        for aad,name,blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32)
            self.known_embeddings.append(emb)
            self.known_meta.append({"aadhaar":aad,"name":name})

    # ---------- REGISTER ----------
    def register(self, img, aadhaar, name):
        faces = self.app.get(img)
        if not faces:
            return False, "No face detected"

        emb = faces[0].embedding.astype(np.float32)

        face_path = f"static/registered_faces/{aadhaar}.jpg"
        aadhaar_path = f"static/aadhaar_photos/{aadhaar}.jpg"

        import cv2
        cv2.imwrite(face_path, img)

        cur = self.conn.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO persons VALUES(?,?,?,?,?)
        """,(aadhaar,name,emb.tobytes(),face_path,aadhaar_path))
        self.conn.commit()

        self._load()
        return True, "Registered successfully"

    # ---------- RECOGNIZE ----------
    def recognize(self, img):
    faces = self.app.get(img)
    results = []

    for f in faces:
        emb = f.embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)

        best_idx = -1
        best_score = -1

        for i, k in enumerate(self.known_embeddings):
            k = k / (np.linalg.norm(k) + 1e-10)
            sim = float(np.dot(emb, k))  # cosine similarity

            if sim > best_score:
                best_score = sim
                best_idx = i

        # 🔥 tuned thresholds
        if best_score > 0.5:
            meta = self.known_meta[best_idx]
            label = meta["name"]
            aadhaar = meta["aadhaar"]
        elif best_score > 0.35:
            label = "Uncertain"
            aadhaar = None
        else:
            label = "Unknown"
            aadhaar = None

        results.append({
            "name": label,
            "aadhaar": aadhaar,
            "confidence": round(best_score, 3)
        })

    return results
