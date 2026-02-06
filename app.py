import os
import torch
from flask import Flask, render_template, request, jsonify
from torchvision import transforms
from PIL import Image
from models.semi_supervised_model import SemiSupervisedPlantModel

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
idx_to_class = None
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    global model, idx_to_class
    model_path = 'models/best_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        class_to_idx = checkpoint['class_to_idx']
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(class_to_idx)
        
        model = SemiSupervisedPlantModel(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully. Classes: {list(idx_to_class.values())}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_image(image_path):
    if model is None:
        return None, None, "Model not loaded"
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, _ = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        predicted_class = idx_to_class[predicted_idx.item()]
        confidence = confidence.item() * 100
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg, gif'}), 400
    
    try:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        predicted_class, confidence, error = predict_image(filepath)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'image': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/classes')
def get_classes():
    if idx_to_class is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({'classes': list(idx_to_class.values())})

@app.route('/health')
def health():
    status = 'ready' if model is not None else 'not_ready'
    return jsonify({'status': status, 'device': str(device)})

if __name__ == '__main__':
    if load_model():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Failed to load model. Exiting...")
