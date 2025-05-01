from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import tempfile

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

# Add zip function to Jinja environment
app.jinja_env.globals.update(zip=zip)

# Load models
plate_detection_model = YOLO('F:/development/python/Number-Plate-Detection/platedetection/training_results/license_plate_detection_yolov8_4_14_25_4_493/weights/best.pt')
char_seg_model = YOLO('F:/development/python/Number-Plate-Detection/segmentation-model/path_project/segmentation_yolov8_4_14_25_4_49/weights/best.pt')

# Character recognition config
CLASS_LABELS = [
    'क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ', 'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा',
    'भे', 'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह', '0', '१', '२', '३', '४', '५',
    '६', '७', '८', '९'
]

class NepaliPlateCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

char_recog_model = NepaliPlateCNN(len(CLASS_LABELS))
char_recog_model.load_state_dict(torch.load("F:/development/python/Number-Plate-Detection/ocr nep/nepali_plate_cnn.pth", map_location='cpu'))
char_recog_model.eval()

def to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

app.jinja_env.filters['to_base64'] = to_base64

def process_file(file_path):
    results = []
    
    if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        frame = cv2.imread(file_path)
        results.extend(process_frame(frame, 0))
    else:
        cap = cv2.VideoCapture(file_path)
        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if frame_number % 5 == 0:
                results.extend(process_frame(frame, frame_number))
            frame_number += 1
        cap.release()
    
    return results

def process_frame(frame, frame_number):
    frame_results = []
    plate_results = plate_detection_model.predict(frame)
    
    for plate in plate_results[0].boxes:
        x1, y1, x2, y2 = map(int, plate.xyxy[0].tolist())
        conf = float(plate.conf[0])
        plate_img = frame[y1:y2, x1:x2]
         # Deskew the plate
        deskewed_plate = deskew_plate(plate_img)
        # Rest of processing uses deskewed_plate instead of plate_img
        char_results = char_seg_model.predict(deskewed_plate)
        char_data = process_characters(deskewed_plate, char_results)
        
        # Convert deskewed plate to base64
        deskewed_pil = Image.fromarray(cv2.cvtColor(deskewed_plate, cv2.COLOR_BGR2RGB))
        deskewed_str = to_base64(deskewed_pil)
        
        char_results = char_seg_model.predict(plate_img)
        print("Character Segmentation Results:", char_results)
        
        char_data = process_characters(plate_img, char_results)
        
        # Convert plate image to base64
        plate_pil = Image.fromarray(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
        plate_img_str = to_base64(plate_pil)
        digital_plate = create_digital_plate(plate_img.shape, char_data)
        digital_plate_str = to_base64(digital_plate)
        frame_results.append({
            'original_plate': plate_img_str,
            'deskewed_plate': deskewed_str,
            'digital_plate': digital_plate_str,
            'frame_number': frame_number,
            'confidence': conf,
            'characters': char_data,
            'final_text': ''.join([c['prediction'] for c in char_data]),
            'plate_dimensions': {
                'width': plate_img.shape[1],
                'height': plate_img.shape[0]
            }
        })
    
    return frame_results


def create_digital_plate(plate_shape, characters):
    # Create white background
    h, w = plate_shape[0], plate_shape[1]
    digital_plate = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(digital_plate)
    
    try:
        # Use Noto Sans Devanagari font (place the .ttf file in your project)
        font = ImageFont.truetype("F:/development/python/Noto_Sans_Devanagari/NotoSansDevanagari-Regular.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Draw each character
    for char in characters:
        # Calculate position
        x_center = (char['x1'] + char['x2']) // 2
        y_center = (char['y1'] + char['y2']) // 2
        
        # Get text size for centering
        text = char['prediction']
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Draw text
        draw.text(
            (x_center - text_w//2, y_center - text_h//2),
            text,
            font=font,
            fill=(0, 0, 255))
    
    return digital_plate
    
    
def process_characters(plate_img, char_results):
    characters = []
    boxes_by_line = {}
    
    for box in char_results[0].boxes:
        y_center = int((box.xyxy[0][1].cpu().numpy() + box.xyxy[0][3].cpu().numpy()) / 2)
        boxes_by_line.setdefault(y_center, []).append(box)
    
    for line_y in sorted(boxes_by_line.keys()):
        line_boxes = sorted(boxes_by_line[line_y], key=lambda b: b.xyxy[0][0].cpu().numpy())
        
        for box in line_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            char_img = plate_img[y1:y2, x1:x2]
            
            # Convert character image to base64
            char_pil = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
            img_str = to_base64(char_pil)
            
            # Character recognition
            input_tensor = preprocess_char_image(char_img)
            with torch.no_grad():
                output = char_recog_model(input_tensor)
            
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            
            characters.append({
                'prediction': CLASS_LABELS[pred.item()],
                'confidence': float(conf),
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'image': img_str
            })
    
    return characters

def preprocess_char_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return torch.tensor(normalized).unsqueeze(0).unsqueeze(0)



def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def deskew_plate(plate_img):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Approximate quadrilateral
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            screenCnt = approx.reshape(4, 2)
            return four_point_transform(plate_img, screenCnt)
    
    # Fallback if no quadrilateral found
    return plate_img



@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename != '':
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(temp_path)
            results = process_file(temp_path)
            return render_template('results.html', results=results)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)