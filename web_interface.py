from flask import Flask, render_template, request, jsonify, send_file
import base64
import io
from PIL import Image

app = Flask(__name__)
system = AadhaarFaceSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        # Get form data
        person_data = {
            'aadhaar_number': request.form['aadhaar_number'],
            'name': request.form['name'],
            'date_of_birth': request.form['dob'],
            'gender': request.form['gender'],
            'address': request.form['address']
        }
        
        # Save uploaded files
        face_image = request.files['face_image']
        aadhaar_photo = request.files['aadhaar_photo']
        
        face_path = f"uploads/faces/{person_data['aadhaar_number']}_face.jpg"
        aadhaar_path = f"uploads/aadhaar/{person_data['aadhaar_number']}_aadhaar.jpg"
        
        face_image.save(face_path)
        aadhaar_photo.save(aadhaar_path)
        
        # Register person
        success, message = system.register_person(face_path, aadhaar_path, person_data)
        
        return jsonify({'success': success, 'message': message})
    
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/recognize', methods=['POST'])
def recognize():
    try:
        # Get image from request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
        
        # Recognize faces
        recognized_faces = system.recognize_face(frame)
        
        return jsonify({'faces': recognized_faces})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)