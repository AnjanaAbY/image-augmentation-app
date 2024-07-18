from flask import Flask, request, render_template, send_file
from flask_uploads import UploadSet, configure_uploads, IMAGES
import cv2
import numpy as np
import os
from io import BytesIO
from zipfile import ZipFile
import albumentations as A

app = Flask(__name__)

# Configure file uploads
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Image upload and augmentation route
@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('photos')
    augmented_images = []

    for file in files:
        filename = photos.save(file)
        filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        augmented_images.extend(augment_image(filepath))

    # Create a zip file of the augmented images
    zip_buffer = create_zip(augmented_images)

    return send_file(zip_buffer, as_attachment=True, download_name='augmented_images.zip', mimetype='application/zip')

def augment_image(filepath):
    image = cv2.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented_images = []

    # Define only the 90 degrees rotation augmentation
    augmentations = [
        A.Rotate(limit=(90, 90), p=1),  # 90 degrees rotation
	A.Rotate(limit=(180, 180), p=1),  # 180 degrees rotation
  	A.Rotate(limit=(270, 270), p=1),  # 270 degrees rotation
	A.HorizontalFlip(p=1),  # Horizontal flip
	A.VerticalFlip(p=1),  # Vertical flip
	A.Compose([
            A.CenterCrop(height=int(image.shape[0]*0.8), width=int(image.shape[1]*0.8), p=1),
            A.Resize(height=image.shape[0], width=image.shape[1], p=1)
        ], p=1),
	A.RandomBrightnessContrast(p=1)
    ]

    # Apply each augmentation independently to the original image
    for aug in augmentations:
        augmented = aug(image=image)['image']
        augmented_images.append(cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

    return augmented_images

def create_zip(images):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'a') as zip_file:
        for i, img in enumerate(images):
            _, img_buffer = cv2.imencode('.jpg', img)
            img_name = f'augmented_{i}.jpg'
            zip_file.writestr(img_name, img_buffer.tobytes())

    zip_buffer.seek(0)
    return zip_buffer

if __name__ == '__main__':
    app.run(debug=True)
