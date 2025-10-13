def main():
    system = AadhaarFaceSystem()
    
    # Example: Register a new person (you would typically get this from a form)
    def register_example_person():
        person_data = {
            'aadhaar_number': '123456789012',
            'name': 'John Doe',
            'date_of_birth': '1990-01-01',
            'gender': 'Male',
            'address': '123 Main St, City, State'
        }
        
        success, message = system.register_person(
            face_image_path='path_to_face_image.jpg',
            aadhaar_photo_path='path_to_aadhaar_photo.jpg',
            person_data=person_data
        )
        print(message)
    
    # Real-time face recognition
    def run_recognition():
        video_capture = cv2.VideoCapture(0)
        
        print("Starting face recognition... Press 'q' to quit, 'd' to show details")
        
        last_recognized_person = None
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Recognize faces
            recognized_faces = system.recognize_face(frame)
            
            # Store last recognized person for details display
            if recognized_faces:
                last_recognized_person = recognized_faces[0]['details']
                last_recognized_person['aadhaar_number'] = recognized_faces[0]['aadhaar_number']
            
            # Display frame
            cv2.imshow('Face Recognition System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d') and last_recognized_person:
                system.display_person_details(last_recognized_person)
        
        video_capture.release()
        cv2.destroyAllWindows()
    
    # Run the system
    run_recognition()

if __name__ == "__main__":
    main()