# ...existing code...
from typing import TYPE_CHECKING

# Let static type checkers / Pylance see the modules while keeping runtime guards
if TYPE_CHECKING:
    import cv2 as cv2  # noqa: F401
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401
    import face_recognition as face_recognition  # noqa: F401

_missing_deps = []
try:
    import cv2
except Exception:
    cv2 = None
    _missing_deps.append('opencv-python (cv2)')

try:
    import numpy as np
except Exception:
    np = None
    _missing_deps.append('numpy')

import sqlite3

try:
    import face_recognition
except Exception:
    face_recognition = None
    _missing_deps.append('face_recognition (requires dlib)')

try:
    import pandas as pd
except Exception:
    pd = None
    _missing_deps.append('pandas')
# ...existing code...

from datetime import datetime
import os
import json

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None
    _missing_deps.append('Pillow')

_install_cmd = (
    "sudo apt update && sudo apt install -y build-essential cmake python3-dev "
    "libatlas-base-dev libjpeg-dev && python3 -m pip install --upgrade pip && "
    "python3 -m pip install numpy opencv-python-headless pillow pandas face_recognition"
)
# ...existing code...

class AadhaarFaceSystem:
    def __init__(self):
        # fail early with actionable message if deps missing
        if _missing_deps:
            raise RuntimeError(
                "Missing Python packages: {}. Install with:\n\n{}".format(
                    ", ".join(_missing_deps), _install_cmd
                )
            )

        self.conn = sqlite3.connect('aadhaar_face_db.db', check_same_thread=False)
        self.create_tables()
        self.known_face_encodings = []
        self.known_face_details = []
        self.load_existing_data()
# ...existing code...
        
    def create_tables(self):
        """Create database tables for storing face and Aadhaar data"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aadhaar_number TEXT UNIQUE,
                name TEXT NOT NULL,
                date_of_birth TEXT,
                gender TEXT,
                address TEXT,
                face_encoding BLOB,
                aadhaar_photo_path TEXT,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                aadhaar_number TEXT,
                recognition_time TIMESTAMP,
                confidence REAL
            )
        ''')
        
        self.conn.commit()
    
    def load_existing_data(self):
        """Load existing face data from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT face_encoding, aadhaar_number, name FROM persons")
        results = cursor.fetchall()
        
        self.known_face_encodings = []
        self.known_face_details = []
        
        for encoding_blob, aadhaar, name in results:
            if encoding_blob:
                try:
                    # stored as raw bytes of float64 array
                    encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                    # ensure 1D numeric array
                    if encoding.size > 0:
                        self.known_face_encodings.append(encoding)
                        self.known_face_details.append({
                            'aadhaar_number': aadhaar,
                            'name': name
                        })
                except Exception:
                    # skip corrupted entries
                    continue

    def save_person(self, aadhaar_number, name, date_of_birth=None, gender=None, address=None, image_path=None):
        """
        Register a person using an image (image_path). Extracts face encoding and stores it.
        Returns True on success, raises ValueError on problems.
        """
        if image_path is None or not os.path.exists(image_path):
            raise ValueError("Valid image_path is required to register person.")
        # load image and compute encoding
        image = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(image)
        if not encs:
            raise ValueError("No face found in the provided image.")
        if len(encs) > 1:
            # optional: choose the largest face or error out
            raise ValueError("Multiple faces found in the image. Provide image with single face.")
        encoding = encs[0].astype(np.float64)
        encoding_blob = encoding.tobytes()
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO persons (aadhaar_number, name, date_of_birth, gender, address, face_encoding, aadhaar_photo_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (aadhaar_number, name, date_of_birth, gender, address, sqlite3.Binary(encoding_blob), image_path))
        self.conn.commit()
        # update in-memory lists
        self.known_face_encodings.append(encoding)
        self.known_face_details.append({'aadhaar_number': aadhaar_number, 'name': name})
        return True

    def recognize(self, frame_bgr, tolerance=0.6, upsample_times=1):
        """
        Recognize faces in a BGR OpenCV frame.
        Returns list of dicts: [{ 'location': (top,right,bottom,left), 'name': str or None, 'aadhaar_number': str or None, 'confidence': float }]
        """
        if not self.known_face_encodings:
            return []

        # convert to RGB as face_recognition expects
        rgb = frame_bgr[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb, number_of_times_to_upsample=upsample_times)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)

        results = []
        for loc, enc in zip(face_locations, face_encodings):
            # compute distances and pick best match
            distances = face_recognition.face_distance(self.known_face_encodings, enc)
            if len(distances) == 0:
                match_index = None
                confidence = 0.0
            else:
                best_idx = int(np.argmin(distances))
                best_dist = float(distances[best_idx])
                is_match = best_dist <= tolerance
                match_index = best_idx if is_match else None
                # confidence: higher when distance smaller; clamp to [0,1]
                confidence = max(0.0, min(1.0, 1.0 - (best_dist / (tolerance if tolerance>0 else 1.0))))
            if match_index is not None:
                detail = self.known_face_details[match_index]
                aadhaar_number = detail.get('aadhaar_number')
                name = detail.get('name')
                # log recognition
                self.log_recognition(aadhaar_number, confidence)
            else:
                aadhaar_number = None
                name = None
            results.append({
                'location': loc,
                'name': name,
                'aadhaar_number': aadhaar_number,
                'confidence': confidence
            })
        return results

    def log_recognition(self, aadhaar_number, confidence):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO recognition_logs (aadhaar_number, recognition_time, confidence)
            VALUES (?, ?, ?)
        ''', (aadhaar_number, datetime.utcnow().isoformat(), float(confidence)))
        self.conn.commit()

    def annotate_image(self, frame_bgr, recognition_results, font_path=None):
        """
        Annotate OpenCV BGR frame with boxes and names. Returns annotated PIL Image.
        """
        # convert to PIL
        rgb = frame_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        # choose font
        try:
            font = ImageFont.truetype(font_path or "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()
        for res in recognition_results:
            top, right, bottom, left = res['location']
            name = res['name'] or "Unknown"
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 255, 0), width=2)
            text = f"{name} ({res['confidence']:.2f})"
            text_width, text_height = draw.textsize(text, font=font)
            draw.rectangle(((left, bottom - text_height - 6), (left + text_width + 6, bottom)), fill=(0, 255, 0))
            draw.text((left + 3, bottom - text_height - 3), text, fill=(0, 0, 0), font=font)
        return pil

    def close(self):
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def __del__(self):
        self.close()
# ...existing code...