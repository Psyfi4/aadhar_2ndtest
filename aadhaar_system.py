#!/usr/bin/env python3
"""
Clean, patched AadhaarFaceSystem
- Robust register_person with debug images & face crops
- recognize_face for per-frame matching
- recognize_live for live-match convenience
- update_person_name, get_person_details utilities
- safe schema migrations to add photo_path columns
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import os
import io
import sqlite3
import time

import numpy as np
import cv2
from PIL import Image

# insightface import may raise ModuleNotFoundError if not installed
from insightface.app import FaceAnalysis


class AadhaarFaceSystem:
    def __init__(self, db_path: str = "aadhaar_face_db.db", debug: bool = True, debug_dir: str = "debug"):
        self.db_path = db_path
        self.debug = bool(debug)
        self.debug_dir = debug_dir

        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
            os.makedirs(os.path.join(self.debug_dir, "crops"), exist_ok=True)
            os.makedirs("static/registered_faces", exist_ok=True)

        # sqlite connection
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        # ensure DB and schema
        self.create_tables()
        self.ensure_schema()

        # initialize InsightFace
        print("[INFO] Initializing InsightFace (buffalo_l).")
        self.app = FaceAnalysis(name="buffalo_l")
        # Use CPU by default (ctx_id=0) — InsightFace will select backend.
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # in-memory caches of embeddings + details
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_details: List[Dict[str, Optional[str]]] = []

        self.load_existing_data()

    # -------------------------
    # Database and schema
    # -------------------------
    def create_tables(self) -> None:
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aadhaar_number TEXT UNIQUE,
                name TEXT,
                date_of_birth TEXT,
                gender TEXT,
                address TEXT,
                embedding BLOB,
                registered_face_path TEXT,
                aadhaar_photo_path TEXT,
                registration_date TIMESTAMP
            );
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aadhaar_number TEXT,
                time TEXT,
                confidence REAL
            );
            """
        )
        self.conn.commit()
        c.close()

    def ensure_schema(self) -> None:
        """
        Add missing columns if DB created with older schema.
        Avoids adding non-constant defaults.
        """
        try:
            c = self.conn.cursor()
            c.execute("PRAGMA table_info(persons);")
            cols = [r[1] for r in c.fetchall()]
            to_add = []
            if "registered_face_path" not in cols:
                to_add.append(("registered_face_path", "TEXT"))
            if "aadhaar_photo_path" not in cols:
                to_add.append(("aadhaar_photo_path", "TEXT"))
            if "registration_date" not in cols:
                to_add.append(("registration_date", "TIMESTAMP"))
            for col_name, col_type in to_add:
                try:
                    c.execute(f"ALTER TABLE persons ADD COLUMN {col_name} {col_type};")
                    print(f"[MIGRATE] Added column persons.{col_name}")
                except Exception as e:
                    # if fails, continue (maybe column exists or sqlite limitation)
                    print(f"[MIGRATE] Failed to add {col_name}: {e}")
            self.conn.commit()
            c.close()
        except Exception as e:
            print("[MIGRATE] ensure_schema failed:", e)

    # -------------------------
    # Load known embeddings
    # -------------------------
    def load_existing_data(self) -> None:
        try:
            c = self.conn.cursor()
            c.execute("SELECT aadhaar_number, name, embedding FROM persons")
            rows = c.fetchall()
            self.known_face_encodings = []
            self.known_face_details = []
            for aadhaar, name, blob in rows:
                if not blob:
                    continue
                try:
                    arr = np.frombuffer(blob, dtype=np.float32)
                    if arr.size == 512:
                        self.known_face_encodings.append(arr)
                        self.known_face_details.append({"aadhaar_number": aadhaar, "name": name})
                except Exception:
                    continue
            c.close()
            print(f"[INFO] Loaded {len(self.known_face_encodings)} registered faces.")
            self.dump_known()
        except Exception as e:
            print("[ERROR] load_existing_data failed:", e)

    def dump_known(self) -> None:
        print(">>> DUMP KNOWN")
        print("count:", len(self.known_face_encodings))
        for i, emb in enumerate(self.known_face_encodings):
            print(i, "shape:", getattr(emb, "shape", None), "dtype:", getattr(emb, "dtype", None))
        print("details:", self.known_face_details[:10])

    # -------------------------
    # Registration helpers
    # -------------------------
    def register_person(self, face_image_path: str, person_data: dict) -> tuple[bool, str]:
        """
        Register a person by image file path.
        Returns (success, message).
        """
        try:
            if not os.path.exists(face_image_path):
                return False, "Face image not found."

            img_bgr = cv2.imread(face_image_path)
            if img_bgr is None:
                return False, "Unable to read image (cv2)."

            ts = int(time.time())
            if self.debug:
                try:
                    cv2.imwrite(os.path.join(self.debug_dir, f"register_input_{ts}.jpg"), img_bgr)
                except Exception:
                    pass

            def try_detect(img_local):
                try:
                    faces = self.app.get(img_local)
                    return faces or []
                except Exception as e:
                    print("[register_person] insightface error:", e)
                    return []

            # Try as-is (BGR)
            faces = try_detect(img_bgr)

            # Try RGB
            if not faces:
                img_rgb = img_bgr[:, :, ::-1].copy()
                faces = try_detect(img_rgb)

            # Try scaling up
            if not faces:
                h, w = img_bgr.shape[:2]
                scale = 1.5
                larger = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
                faces = try_detect(larger)
                if faces and self.debug:
                    try:
                        cv2.imwrite(os.path.join(self.debug_dir, f"register_input_scaled_{ts}.jpg"), larger)
                    except Exception:
                        pass

            # Try equalization
            if not faces:
                try:
                    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    l = cv2.equalizeHist(l)
                    lab_eq = cv2.merge((l, a, b))
                    eq_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
                    faces = try_detect(eq_bgr)
                    if faces and self.debug:
                        try:
                            cv2.imwrite(os.path.join(self.debug_dir, f"register_input_eq_{ts}.jpg"), eq_bgr)
                        except Exception:
                            pass
                except Exception:
                    pass

            if not faces:
                # save debug overlay
                if self.debug:
                    try:
                        dbg = img_bgr.copy()
                        h, w = dbg.shape[:2]
                        cv2.rectangle(dbg, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 2)
                        cv2.putText(dbg, "No face detected - register", (10, max(20, h - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imwrite(os.path.join(self.debug_dir, f"register_no_face_{ts}.jpg"), dbg)
                    except Exception:
                        pass
                return False, "No face detected in image."

            if len(faces) > 1:
                if self.debug:
                    try:
                        dbg2 = img_bgr.copy()
                        for i, f in enumerate(faces):
                            try:
                                x1, y1, x2, y2 = map(int, f.bbox)
                                cv2.rectangle(dbg2, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(dbg2, f"#{i+1}", (x1, max(10, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                            (0, 0, 255), 2)
                            except Exception:
                                pass
                        cv2.imwrite(os.path.join(self.debug_dir, f"register_multiple_{ts}.jpg"), dbg2)
                    except Exception:
                        pass
                return False, "Multiple faces detected; provide a single-face image."

            face_obj = faces[0]
            try:
                emb = face_obj.embedding.astype(np.float32).reshape(-1)
            except Exception:
                return False, "Failed to extract embedding."

            if emb.size != 512:
                return False, "Unexpected embedding size."

            blob = emb.tobytes()

            # Save face crop and registered_face_path
            try:
                x1, y1, x2, y2 = map(int, face_obj.bbox)
                crop = img_bgr[y1:y2, x1:x2]
                reg_path = os.path.join("static", "registered_faces", f"{person_data.get('aadhaar_number')}_{ts}.jpg")
                cv2.imwrite(reg_path, crop)
            except Exception:
                reg_path = None

            try:
                c = self.conn.cursor()
                c.execute(
                    """
                    INSERT OR REPLACE INTO persons
                    (aadhaar_number, name, date_of_birth, gender, address, embedding, registered_face_path, registration_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        person_data.get("aadhaar_number"),
                        person_data.get("name"),
                        person_data.get("date_of_birth"),
                        person_data.get("gender"),
                        person_data.get("address"),
                        blob,
                        reg_path,
                        datetime.utcnow().isoformat(),
                    ),
                )
                self.conn.commit()
                c.close()
            except Exception as e:
                return False, f"DB save error: {e}"

            # update caches
            self.known_face_encodings.append(emb)
            self.known_face_details.append({"aadhaar_number": person_data.get("aadhaar_number"),
                                            "name": person_data.get("name")})

            # save a facecrop to debug/crops
            if self.debug and reg_path:
                try:
                    crop_ts = int(time.time())
                    dbg_crop_path = os.path.join(self.debug_dir, "crops", os.path.basename(reg_path))
                    if not os.path.exists(dbg_crop_path):
                        cv2.imwrite(dbg_crop_path, crop)
                except Exception:
                    pass

            return True, "Person registered successfully."
        except Exception as e:
            print("[register_person] unexpected:", e)
            return False, f"Internal error: {e}"
        
        h, w = img_bgr.shape[:2]
        if max(h, w) < 480:
            scale = 480 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)))
            print("[DEBUG] Upscaled tiny image for detection:", img_bgr.shape)


    # -------------------------
    # Register from frame (numpy array)
    # -------------------------
    def register_face(self, frame: np.ndarray, aadhaar_number: str, name: str, extra: dict = None) -> Dict[str, Any]:
        """
        Accepts a numpy image (RGB or BGR) and registers the face (saves crop and embedding).
        Returns a dict with success and info.
        """
        if frame is None:
            return {"success": False, "error": "No frame provided"}

        try:
            # ensure BGR for insightface
            arr = np.asarray(frame)
            if arr.shape[2] == 3:
                bgr = arr[:, :, ::-1]
            else:
                bgr = arr[:, :, :3]

            faces = self.app.get(bgr)
            if not faces:
                return {"success": False, "error": "No face detected"}

            face = faces[0]
            emb = face.embedding.astype(np.float32).reshape(-1)
            if emb.size != 512:
                return {"success": False, "error": "invalid embedding"}

            ts = int(time.time())
            try:
                x1, y1, x2, y2 = map(int, face.bbox)
                crop = bgr[y1:y2, x1:x2]
                save_path = os.path.join("static", "registered_faces", f"{aadhaar_number}_{ts}.jpg")
                cv2.imwrite(save_path, crop)
            except Exception:
                save_path = None

            c = self.conn.cursor()
            c.execute(
                """
                INSERT OR REPLACE INTO persons (aadhaar_number, name, date_of_birth, gender, address, embedding, registered_face_path, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    aadhaar_number,
                    name,
                    (extra or {}).get("date_of_birth"),
                    (extra or {}).get("gender"),
                    (extra or {}).get("address"),
                    emb.tobytes(),
                    save_path,
                    datetime.utcnow().isoformat(),
                ),
            )
            self.conn.commit()
            c.close()

            # refresh caches
            self.load_existing_data()
            return {"success": True, "face_crop": save_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # -------------------------
    # Recognition (per-frame)
    # -------------------------
    def recognize_face(self, frame: Any) -> List[Dict[str, Any]]:
        """
        Accepts a numpy image-like frame (RGB or BGR). Returns list of matches dicts.
        """
        try:
            frame_np = np.asarray(frame)
        except Exception:
            return []

        if getattr(frame_np, "ndim", None) != 3:
            return []

        # save debug frame
        if self.debug:
            try:
                ts = int(time.time())
                bgr_save = frame_np[:, :, :3][:, :, ::-1]
                cv2.imwrite(os.path.join(self.debug_dir, f"last_recognize_{ts}.jpg"), bgr_save)
            except Exception:
                pass

        # ensure BGR for insightface
        if frame_np.shape[2] == 3:
            bgr = frame_np[:, :, ::-1]
        else:
            bgr = frame_np[:, :, :3]

        try:
            faces = self.app.get(bgr)
        except Exception as e:
            print("[recognize_face] app.get failed:", e)
            return []

        if not faces:
            return []

        results: List[Dict[str, Any]] = []
        for f in faces:
            try:
                emb = f.embedding.astype(np.float32).reshape(-1)
            except Exception:
                continue

            sims = []
            confs = []
            for known in self.known_face_encodings:
                a = emb / (np.linalg.norm(emb) + 1e-10)
                b = known / (np.linalg.norm(known) + 1e-10)
                sim = float(np.dot(a, b))
                conf = (sim + 1.0) / 2.0
                conf = max(0.0, min(1.0, conf))
                sims.append(sim)
                confs.append(conf)

            best_idx = int(np.argmax(confs)) if confs else None
            best_conf = float(confs[best_idx]) if best_idx is not None else 0.0

            try:
                x1, y1, x2, y2 = map(int, f.bbox)
                top, right, bottom, left = y1, x2, y2, x1
            except Exception:
                top = right = bottom = left = 0

            MATCH_THRESHOLD = 0.44
            POSSIBLE_MARGIN = 0.02

            name = None
            aadhaar = None
            matched = False
            unconfirmed = False

            if best_idx is not None:
                if best_conf >= MATCH_THRESHOLD:
                    detail = self.known_face_details[best_idx]
                    name = detail.get("name")
                    aadhaar = detail.get("aadhaar_number")
                    matched = True
                    try:
                        self.log_recognition(aadhaar, float(best_conf))
                    except Exception:
                        pass
                elif best_conf >= (MATCH_THRESHOLD - POSSIBLE_MARGIN):
                    detail = self.known_face_details[best_idx]
                    name = detail.get("name")
                    aadhaar = detail.get("aadhaar_number")
                    unconfirmed = True

            # save crop if debug
            if self.debug and (bottom - top) > 10 and (right - left) > 10:
                try:
                    h, w = frame_np.shape[:2]
                    t = max(0, min(h - 1, top))
                    btm = max(0, min(h, bottom))
                    l = max(0, min(w - 1, left))
                    r = max(0, min(w, right))
                    crop = frame_np[t:btm, l:r, :3]
                    if crop.size:
                        save_crop = crop[:, :, ::-1]
                        crop_path = os.path.join(self.debug_dir, "crops", f"{aadhaar or 'unknown'}_{int(time.time())}.jpg")
                        cv2.imwrite(crop_path, save_crop)
                except Exception:
                    pass

            details = None
            try:
                details = self.get_person_details(aadhaar) if aadhaar else None
            except Exception:
                details = None

            results.append({
                "location": (top, right, bottom, left),
                "name": name,
                "aadhaar_number": aadhaar,
                "confidence": float(best_conf),
                "matched": matched,
                "unconfirmed": unconfirmed,
                "details": details
            })

        return results

    # -------------------------
    # Recognize live convenience (returns best_person dict or None)
    # -------------------------
    def recognize_live(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Simpler live recognition - returns best matched person dict or None.
        Uses Euclidean distance on stored embeddings.
        """
        try:
            arr = np.asarray(frame)
            if arr.shape[2] == 3:
                bgr = arr[:, :, ::-1]
            else:
                bgr = arr[:, :, :3]
            faces = self.app.get(bgr)
            if not faces:
                return None
            f = faces[0]
            emb = f.embedding.astype(np.float32).reshape(-1)

            cur = self.conn.cursor()
            cur.execute("SELECT aadhaar_number, name, registered_face_path, embedding, aadhaar_photo_path, date_of_birth, gender, address FROM persons")
            rows = cur.fetchall()
            cur.close()
            best = None
            best_dist = float("inf")
            for row in rows:
                try:
                    stored_emb = np.frombuffer(row[3], dtype=np.float32)
                    dist = np.linalg.norm(stored_emb - emb)
                    if dist < best_dist:
                        best_dist = dist
                        best = {
                            "aadhaar_number": row[0],
                            "name": row[1],
                            "registered_face": row[2],
                            "aadhaar_photo": row[4],
                            "date_of_birth": row[5],
                            "gender": row[6],
                            "address": row[7],
                            "distance": float(dist)
                        }
                except Exception:
                    continue
            return best
        except Exception as e:
            print("[recognize_live] error:", e)
            return None

    # -------------------------
    # Update utilities, logging
    # -------------------------
    def update_person_name(self, aadhaar_number: str, name: str) -> bool:
        if not aadhaar_number:
            return False
        try:
            cur = self.conn.cursor()
            cur.execute("UPDATE persons SET name = ? WHERE aadhaar_number = ?", (name, aadhaar_number))
            self.conn.commit()
            cur.close()
            self.load_existing_data()
            return True
        except Exception as e:
            print("[ERROR] update_person_name failed:", e)
            return False

    def log_recognition(self, aadhaar_number: str, confidence: float) -> None:
        if not aadhaar_number:
            return
        try:
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO recognition_logs (aadhaar_number, time, confidence) VALUES (?, ?, ?)",
                (aadhaar_number, datetime.utcnow().isoformat(), float(confidence)),
            )
            self.conn.commit()
            c.close()
        except Exception as e:
            print("[ERROR] log_recognition failed:", e)

    def get_person_details(self, aadhaar_number: str) -> Optional[Dict[str, Optional[str]]]:
        if not aadhaar_number:
            return None
        try:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT aadhaar_number, name, date_of_birth, gender, address, registered_face_path, aadhaar_photo_path, registration_date FROM persons WHERE aadhaar_number = ?",
                (aadhaar_number,),
            )
            row = cur.fetchone()
            cur.close()
            if row:
                return {
                    "aadhaar_number": row[0],
                    "name": row[1],
                    "date_of_birth": row[2],
                    "gender": row[3],
                    "address": row[4],
                    "registered_face_path": row[5],
                    "aadhaar_photo_path": row[6],
                    "registration_date": row[7],
                }
        except Exception:
            return None
        return None

    # -------------------------
    # Close DB
    # -------------------------
    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    s = AadhaarFaceSystem(debug=True)
    s.dump_known()
