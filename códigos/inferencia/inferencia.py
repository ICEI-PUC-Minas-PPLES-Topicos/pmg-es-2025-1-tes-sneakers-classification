import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
import json
import io

app = Flask(__name__)

# Load class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Define GoogLeNet model
def create_googlenet(num_classes=50):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=False, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model 

# Load trained model
model = create_googlenet(len(class_labels) + 1)
model.load_state_dict(torch.load('./googlenet_sneakers.pth', weights_only=True, map_location=torch.device('cpu')))
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Process image
        image = request.files['image'].read()
        image = Image.open(io.BytesIO(image)).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
        
        return jsonify({
            'class_id': class_id,
            'class_name': class_labels[class_id],
            'confidence': torch.nn.functional.softmax(outputs, dim=1)[0][class_id].item()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
