import cv2
import numpy as np
import sqlite3
import face_recognition
import pandas as pd
from datetime import datetime
import os
import json
from PIL import Image, ImageDraw, ImageFont

class AadhaarFaceSystem:
    def __init__(self):
        self.conn = sqlite3.connect('aadhaar_face_db.db', check_same_thread=False)
        self.create_tables()
        self.known_face_encodings = []
        self.known_face_details = []
        self.load_existing_data()
        
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
                encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                self.known_face_encodings.append(encoding)
                self.known_face_details.append({
                    'aadhaar_number': aadhaar,
                    'name': name
                })