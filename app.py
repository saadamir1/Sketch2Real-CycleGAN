"""
UI for image conversion operations... Sketch to Image & Image to Sketch
"""
from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import cv2
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Setup the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Encoder (downsampling)
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Bottleneck
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(),
            
            # Decoder (upsampling)
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# Load models
G = Generator().to(device)  # Sketch to Real
F = Generator().to(device)  # Real to Sketch
G.load_state_dict(torch.load('generator_G.pth', map_location=device))
F.load_state_dict(torch.load('generator_F.pth', map_location=device))
G.eval()
F.eval()

# Image processing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def tensor_to_base64(tensor):
    tensor = (tensor.cpu().clone().detach() + 1) / 2
    pil_image = transforms.ToPILImage()(tensor.squeeze(0))
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/jpeg;base64,{img_str}"

def is_sketch(img_array):
    # Calculate features that help identify sketches
    blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 100)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    white_ratio = np.sum(img_array > 230) / (img_array.shape[0] * img_array.shape[1])
    avg_pixel = np.mean(img_array)
    
    # Sketches normally, have high edge density and more white areas...
    return (edge_density > 0.05 and white_ratio > 0.6) or (avg_pixel > 200 and edge_density > 0.03)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    # Get image from file upload or camera
    if 'file' in request.files and request.files['file'].filename:
        img = Image.open(request.files['file']).convert('L')
    elif 'camera_data' in request.form:
        camera_data = request.form['camera_data'].split('base64,')[1] if 'base64,' in request.form['camera_data'] else request.form['camera_data']
        img = Image.open(BytesIO(base64.b64decode(camera_data))).convert('L')
    else:
        return redirect(url_for('home'))
    
    # Process image
    img_array = np.array(img)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Determine image type and convert
    is_sketch_img = is_sketch(img_array)
    with torch.no_grad():
        if is_sketch_img:
            generated = G(img_tensor)  # Sketch to real
            img_type, conversion_type = "sketch", "real"
        else:
            generated = F(img_tensor)  # Real to sketch
            img_type, conversion_type = "real", "sketch"
    
    # Prepare images for display
    original_base64 = "data:image/jpeg;base64," + base64.b64encode(
        BytesIO(cv2.imencode('.jpg', img_array)[1]).getvalue()
    ).decode('utf-8')
    
    generated_base64 = tensor_to_base64(generated)
    
    return render_template('result.html', 
                          original_img=original_base64,
                          generated_img=generated_base64,
                          input_type=img_type,
                          output_type=conversion_type)

if __name__ == '__main__':
    app.run(debug=True)