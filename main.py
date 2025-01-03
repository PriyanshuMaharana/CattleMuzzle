import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import logging
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
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    INPUT_SHAPE = (71, 71, 3)

app.config.from_object(Config)

class TFLiteModelService:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = self.load_model()

    def load_model(self):
        """Load the TFLite model."""
        try:
            interpreter = tf.lite.Interpreter(model_path=self.model_path)
            interpreter.allocate_tensors()
            self.input_details = interpreter.get_input_details()
            self.output_details = interpreter.get_output_details()
            logger.info("Model loaded successfully")
            return interpreter
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
        features = self.interpreter.get_tensor(self.output_details[0]['index'])
        return features.flatten()[:256]

# Initialize the TFLite model service
tflite_model_service = TFLiteModelService(model_path="model.tflite")

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
                img_array = tflite_model_service.preprocess_image(tmp_file.name)
                features = tflite_model_service.extract_features(img_array)
                
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
