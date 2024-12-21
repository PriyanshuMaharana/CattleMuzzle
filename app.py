from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import logging
import firebase_admin
from firebase_admin import credentials, ml
import tempfile
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
class Config:
    FIREBASE_MODEL_NAME = os.getenv('FIREBASE_MODEL_NAME', 'MAIN_MUZZLE')
    FIREBASE_CREDENTIALS_PATH = 'path/to/your/firebase-credentials.json'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    INPUT_SHAPE = (71, 71, 3)

app.config.from_object(Config)

class FirebaseMLService:
    def __init__(self):
        self._init_firebase()
        self.interpreter = None
        
    def _init_firebase(self):
        """Initialize Firebase with credentials"""
        if not firebase_admin._apps:
            cred = credentials.Certificate(app.config['FIREBASE_CREDENTIALS_PATH'])
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized successfully")
    
    def load_model(self):
        """Load model from Firebase ML"""
        try:
            # Get model reference
            model = ml.get_model(app.config['FIREBASE_MODEL_NAME'])
            
            # Create temporary file for model
            with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_model_file:
                model_path = tmp_model_file.name
                
            # Download model
            model.download_to_file(model_path)
            
            # Load the TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info("Model loaded successfully")
            
            # Clean up
            os.unlink(model_path)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_image(self, img_path):
        """Preprocess image for model input"""
        img = image.load_img(img_path, target_size=app.config['INPUT_SHAPE'][:2])
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return (img_array / 255.0).astype(np.float32)
    
    def extract_features(self, img_array):
        """Extract features using the TFLite model"""
        self.interpreter.set_tensor(self.input_details[0]['index'], img_array)
        self.interpreter.invoke()
        
        # Get the output from the penultimate layer
        features = self.interpreter.get_tensor(self.output_details[-2]['index'])
        return features.flatten()[:256]

# Initialize Firebase ML service
firebase_service = FirebaseMLService()

@app.before_first_request
def initialize_model():
    """Initialize the model before first request"""
    try:
        firebase_service.load_model()
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """Extract features from uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file'}), 400
        
        # Create temporary file for the image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            file.save(tmp_file.name)
            
            try:
                img_array = firebase_service.preprocess_image(tmp_file.name)
                features = firebase_service.extract_features(img_array)
                
                return jsonify({
                    'message': 'Features extracted successfully',
                    'features': features.tolist()
                })
            
            finally:
                # Clean up
                os.unlink(tmp_file.name)
                
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
