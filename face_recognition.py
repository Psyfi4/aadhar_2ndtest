    def recognize_face(self, frame, tolerance=0.6):
        """
        Recognize faces in real-time video frame
        
        Args:
            frame: Video frame from camera
            tolerance: Face recognition tolerance (lower is more strict)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and encodings in current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        recognized_faces = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=tolerance
            )
            
            name = "Unknown"
            aadhaar_number = None
            confidence = 0.0
            
            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_details[first_match_index]['name']
                aadhaar_number = self.known_face_details[first_match_index]['aadhaar_number']
                
                # Calculate confidence based on distance
                face_distances = face_recognition.face_distance(
                    self.known_face_encodings, face_encoding
                )
                confidence = 1 - face_distances[first_match_index]
                
                # Log recognition
                self.log_recognition(aadhaar_number, confidence)
                
                # Get full person details
                person_details = self.get_person_details(aadhaar_number)
                
                recognized_faces.append({
                    'location': (top, right, bottom, left),
                    'name': name,
                    'aadhaar_number': aadhaar_number,
                    'confidence': confidence,
                    'details': person_details
                })
            
            # Draw bounding box and info
            self.draw_face_info(frame, top, right, bottom, left, name, confidence)
        
        return recognized_faces
    
    def get_person_details(self, aadhaar_number):
        """Retrieve complete person details from database"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT name, date_of_birth, gender, address, aadhaar_photo_path 
            FROM persons WHERE aadhaar_number = ?
        ''', (aadhaar_number,))
        
        result = cursor.fetchone()
        if result:
            return {
                'name': result[0],
                'date_of_birth': result[1],
                'gender': result[2],
                'address': result[3],
                'aadhaar_photo_path': result[4]
            }
        return None
    
    def log_recognition(self, aadhaar_number, confidence):
        """Log recognition events"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO recognition_logs (aadhaar_number, recognition_time, confidence)
            VALUES (?, ?, ?)
        ''', (aadhaar_number, datetime.now(), confidence))
        self.conn.commit()