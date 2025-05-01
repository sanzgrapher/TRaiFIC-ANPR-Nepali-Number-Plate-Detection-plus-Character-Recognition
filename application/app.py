# --- Imports ---
import flask
from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import tempfile
import math
import time
import logging
import shutil

# --- Configuration ---
# !!! IMPORTANT: Replace these placeholder paths with your actual model paths !!!
PLATE_MODEL_PATH = 'F:/development/python/Number-Plate-Detection/platedetection/training_results/license_plate_detection_yolov8_4_14_25_4_493/weights/best.pt'
CHAR_SEG_MODEL_PATH = 'F:/development/python/Number-Plate-Detection/segmentation-model/path_project/segmentation_yolov8_4_14_25_4_49/weights/best.pt'
CHAR_REC_MODEL_PATH = "F:/development/python/Number-Plate-Detection/ocr nep/nepali_plate_cnn.pth"
# !!! IMPORTANT: Ensure this font file exists or change the path !!!
FONT_PATH = "F:/development/python/Noto_Sans_Devanagari/NotoSansDevanagari-Regular.ttf"

# Character recognition config
CLASS_LABELS = [
    'क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ', 'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा',
    'भे', 'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह', '0', '१', '२', '३', '४', '५',
    '६', '७', '८', '९'
]
# Check consistency with standalone script (e.g., '०' vs '0') - Using digits from your Flask code
# CLASS_LABELS = [
#     'क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ', 'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा',
#     'भे', 'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह', '०', '१', '२', '३', '४', '५',
#     '६', '७', '८', '९'
# ]
NUM_CLASSES = len(CLASS_LABELS)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Setup ---
app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_NAME = 'anpr_uploads'
UPLOAD_FOLDER_PATH = os.path.join(APP_ROOT, UPLOAD_FOLDER_NAME)
os.makedirs(UPLOAD_FOLDER_PATH, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
app.secret_key = 'your_very_secret_key_change_me'

logging.info(f"Using upload folder: {app.config['UPLOAD_FOLDER']}")
app.jinja_env.globals.update(zip=zip)

# --- Check Model and Font Files ---
# (Keep the checks as before)
if not os.path.exists(PLATE_MODEL_PATH): logging.error(f"Plate detection model not found: {PLATE_MODEL_PATH}")
if not os.path.exists(CHAR_SEG_MODEL_PATH): logging.error(f"Character segmentation model not found: {CHAR_SEG_MODEL_PATH}")
if not os.path.exists(CHAR_REC_MODEL_PATH): logging.error(f"Character recognition model not found: {CHAR_REC_MODEL_PATH}")
if not os.path.exists(FONT_PATH):
    logging.warning(f"Font file not found: {FONT_PATH}. Using default font.")
    FONT_PATH = None

# --- Model Definition ---
# Using the definition from your standalone script for consistency
class NepaliPlateCNN(nn.Module):
    def __init__(self, num_classes):
        super(NepaliPlateCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Correct flatten size calculation: Output of pool3 is 128 x (32/2/2/2) x (32/2/2/2) = 128 x 4 x 4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        # Added Dropout like in previous Flask version (often good for regularization)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout
        x = self.fc2(x)
        return x

# --- Load Models ---
plate_detection_model = None
char_seg_model = None
char_recog_model = None
device = None

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if os.path.exists(PLATE_MODEL_PATH):
        plate_detection_model = YOLO(PLATE_MODEL_PATH)
        logging.info(f"Loaded Plate Detection Model from: {PLATE_MODEL_PATH}")
    else: logging.error("Plate detection model file missing.")

    if os.path.exists(CHAR_SEG_MODEL_PATH):
        char_seg_model = YOLO(CHAR_SEG_MODEL_PATH)
        logging.info(f"Loaded Character Segmentation Model from: {CHAR_SEG_MODEL_PATH}")
    else: logging.error("Character segmentation model file missing.")

    if os.path.exists(CHAR_REC_MODEL_PATH):
        # Use the consistent model definition
        char_recog_model = NepaliPlateCNN(NUM_CLASSES)
        char_recog_model.load_state_dict(torch.load(CHAR_REC_MODEL_PATH, map_location=device))
        char_recog_model.to(device)
        char_recog_model.eval()
        logging.info(f"Loaded Character Recognition Model from: {CHAR_REC_MODEL_PATH}")
    else: logging.error("Character recognition model file missing.")

except Exception as e:
    logging.error(f"Error loading models: {e}", exc_info=True)

# --- Utility Functions ---
def to_base64(image_pil):
    """Converts a PIL Image to a base64 string."""
    if not isinstance(image_pil, Image.Image):
        logging.warning("to_base64 received non-PIL image, attempting conversion.")
        try:
            if isinstance(image_pil, np.ndarray):
                image_pil = Image.fromarray(cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.new('RGB', (50, 20), color = 'red')
                ImageDraw.Draw(image_pil).text((5,5), "Error", fill="white")
        except Exception as conv_err:
             logging.error(f"Error converting input to PIL image in to_base64: {conv_err}")
             image_pil = Image.new('RGB', (50, 20), color = 'red')
             ImageDraw.Draw(image_pil).text((5,5), "Error", fill="white")

    buffered = BytesIO()
    try:
        image_pil.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error saving image to buffer for base64 encoding: {e}")
        return None

app.jinja_env.filters['to_base64'] = to_base64

# ============================================================================
# == CORRECTED CHARACTER PREPROCESSING FUNCTION TO MATCH TRAINING/STANDALONE ==
# ============================================================================
def preprocess_char_image(image_cv):
    """
    Preprocesses a single character image (OpenCV format) for recognition.
    MUST MATCH the preprocessing used during model training.
    """
    if image_cv is None or image_cv.size == 0:
        logging.warning("preprocess_char_image received empty image.")
        return None

    # 1. Convert to Grayscale
    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    elif len(image_cv.shape) == 2:
        gray = image_cv # Already grayscale
    else:
        logging.warning(f"Unexpected image format in preprocess_char_image: shape={image_cv.shape}")
        return None

    # --- Critical Steps Matching Training ---
    # 2. Direct Resize to 32x32 (potentially distorting aspect ratio)
    try:
        # Use INTER_AREA for shrinking, INTER_LINEAR or INTER_CUBIC for enlarging generally works well
        interpolation_method = cv2.INTER_AREA if gray.shape[0] > 32 or gray.shape[1] > 32 else cv2.INTER_LINEAR
        resized = cv2.resize(gray, (32, 32), interpolation=interpolation_method)
    except cv2.error as resize_err:
         logging.error(f"OpenCV error during direct resize: {resize_err}. Image shape: {gray.shape}")
         return None

    # 3. Normalize to [-1, 1] range
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    # --- End Critical Steps ---

    # 4. Convert to PyTorch Tensor and add Batch/Channel dimensions
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # 5. Move to the correct device
    return tensor.to(device)
# ============================================================================
# == END OF CORRECTED PREPROCESSING FUNCTION                                ==
# ============================================================================


# --- Perspective Correction / Deskewing (Keep as before) ---
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
    try:
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        if maxWidth <= 0 or maxHeight <=0: return None
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    except Exception as e:
        logging.error(f"Error during four_point_transform: {e}", exc_info=True)
        return None

def deskew_plate(plate_img):
    if plate_img is None or plate_img.size == 0: return plate_img
    min_height, min_width = 15, 30
    if plate_img.shape[0] < min_height or plate_img.shape[1] < min_width: return plate_img
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return plate_img
        possible_plates = []
        h_img, w_img = plate_img.shape[:2]
        plate_area = h_img * w_img
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0.05 * plate_area and area < 0.95 * plate_area:
                 peri = cv2.arcLength(cnt, True)
                 approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                 if len(approx) == 4: possible_plates.append(approx)
        if not possible_plates:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect); box = np.intp(box)
            rect_width, rect_height = rect[1]
            aspect_ratio = max(rect_width, rect_height) / max(1, min(rect_width, rect_height))
            if rect_width > min_width/2 and rect_height > min_height/2 and aspect_ratio < 10:
                 warped = four_point_transform(plate_img, box.astype("float32"))
                 if warped is not None and warped.shape[0] >= min_height and warped.shape[1] >= min_width: return warped
            return plate_img # Fallback minAreaRect failed or didn't look right
        largest_quad = max(possible_plates, key=cv2.contourArea)
        screenCnt = largest_quad.reshape(4, 2)
        warped = four_point_transform(plate_img, screenCnt.astype("float32"))
        if warped is not None and warped.shape[0] >= min_height and warped.shape[1] >= min_width: return warped
        return plate_img # Fallback approxPoly failed
    except Exception as e:
        logging.error(f"Error during deskewing: {e}", exc_info=True)
        return plate_img

# --- Character Processing and Ordering (Keep as before) ---
def process_and_order_characters(deskewed_plate, char_results):
    if char_recog_model is None:
         logging.error("Character recognition model not loaded. Cannot process characters.")
         return []
    characters = []
    if not char_results or not hasattr(char_results[0], 'boxes') or not char_results[0].boxes:
        return []
    raw_chars = []
    h_plate, w_plate = deskewed_plate.shape[:2]
    for box in char_results[0].boxes:
        conf = float(box.conf[0]) if box.conf is not None else 1.0
        if conf < 0.3: continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_plate, x2), min(h_plate, y2)
        if x1 >= x2 or y1 >= y2: continue

        char_img_cv = deskewed_plate[y1:y2, x1:x2]
        input_tensor = preprocess_char_image(char_img_cv) # USE CORRECTED PREPROCESSING
        if input_tensor is None: continue

        try:
            with torch.no_grad():
                output = char_recog_model(input_tensor)
                probs = F.softmax(output, dim=1)
                pred_conf, pred_idx = torch.max(probs, 1)
                if pred_idx.item() < len(CLASS_LABELS):
                    pred_label = CLASS_LABELS[pred_idx.item()]
                else:
                    logging.error(f"Prediction index {pred_idx.item()} out of range.")
                    pred_label = '?'
                pred_conf = float(pred_conf.item())

            raw_chars.append({
                'prediction': pred_label, 'confidence': pred_conf,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
                'char_img_cv': char_img_cv
            })
        except Exception as rec_err:
             logging.error(f"Error during character recognition step: {rec_err}", exc_info=True)
             continue

    if not raw_chars: return []

    # --- Line Detection and Sorting ---
    valid_heights = [(c['y2'] - c['y1']) for c in raw_chars if (c['y2'] - c['y1']) > 0]
    avg_char_height = np.mean(valid_heights) if valid_heights else h_plate / 4
    if avg_char_height <= 0: avg_char_height = 10 # Prevent division by zero or negative heights

    y_centers = sorted([c['cy'] for c in raw_chars])
    num_lines = 1
    line_threshold_y = h_plate

    if len(y_centers) > 1:
        vertical_spread = y_centers[-1] - y_centers[0]
        if vertical_spread > avg_char_height * 1.1:
            max_gap = 0; split_index = -1
            for i in range(len(y_centers) - 1):
                gap = y_centers[i+1] - y_centers[i]
                if gap > max_gap: max_gap = gap; split_index = i
            if split_index != -1 and max_gap > avg_char_height * 0.5:
                num_lines = 2
                line_threshold_y = (y_centers[split_index] + y_centers[split_index + 1]) / 2
                logging.debug(f"Detected 2 lines. Threshold: {line_threshold_y:.1f}")
            else: logging.debug("Spread suggests >1 line, but no clear gap. Treating as 1 line.")
        else: logging.debug("Vertical spread suggests 1 line.")

    line1_chars, line2_chars = [], []
    if num_lines == 1: line1_chars = raw_chars
    else:
        for char_data in raw_chars:
            if char_data['cy'] < line_threshold_y: line1_chars.append(char_data)
            else: line2_chars.append(char_data)

    line1_chars.sort(key=lambda c: c['cx'])
    line2_chars.sort(key=lambda c: c['cx'])
    sorted_characters_raw = line1_chars + line2_chars

    final_char_list = []
    for char_data in sorted_characters_raw:
        img_str = None
        try:
            char_pil = Image.fromarray(cv2.cvtColor(char_data['char_img_cv'], cv2.COLOR_BGR2RGB))
            img_str = to_base64(char_pil)
        except Exception as img_conv_err: logging.error(f"Error converting char img to base64: {img_conv_err}")
        final_char_list.append({
            'prediction': char_data['prediction'], 'confidence': char_data['confidence'],
            'x1': char_data['x1'], 'y1': char_data['y1'], 'x2': char_data['x2'], 'y2': char_data['y2'],
            'image': img_str
        })
    return final_char_list


# --- Digital Plate Creation (Keep as before) ---
def create_digital_plate(plate_shape, characters):
    h, w = plate_shape[0], plate_shape[1]
    if w <= 0 or h <= 0:
        digital_plate = Image.new('RGB', (150, 50), (200, 200, 200))
        draw = ImageDraw.Draw(digital_plate); draw.text((5, 5), "Invalid Size", fill="red")
        return digital_plate

    digital_plate = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(digital_plate)
    font = None; font_size = 40

    if characters:
        valid_heights = [(c['y2'] - c['y1']) for c in characters if (c['y2'] - c['y1']) > 0]
        if valid_heights: avg_char_h = np.mean(valid_heights)
        else: avg_char_h = h / 3 # Guess if no valid heights
        font_size = max(10, min(int(h * 0.8), int(avg_char_h * 0.8)))
    else: font_size = max(10, min(int(h*0.8), 30))

    try:
        if FONT_PATH and os.path.exists(FONT_PATH): font = ImageFont.truetype(FONT_PATH, font_size)
        else: font = ImageFont.load_default() # Basic fallback
    except Exception as font_err:
        logging.error(f"Error loading font: {font_err}. Using basic default.")
        try: font = ImageFont.load_default()
        except Exception: font = None

    if font is None:
         logging.error("Cannot draw text on digital plate: No font loaded.")
         for char in characters: draw.rectangle([char['x1'], char['y1'], char['x2'], char['y2']], outline="red", width=2)
         return digital_plate

    for char in characters:
        text = char['prediction']; x1, y1, x2, y2 = char['x1'], char['y1'], char['x2'], char['y2']
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        try:
            if hasattr(draw, 'textbbox'): draw.text((x_center, y_center), text, font=font, fill=(0, 0, 0), anchor="ms")
            else: text_w, text_h = draw.textsize(text, font=font); draw.text((x_center - text_w / 2, y_center - text_h / 2), text, font=font, fill=(0, 0, 0))
        except Exception as draw_err:
             logging.error(f"Error drawing text '{text}': {draw_err}")
             draw.rectangle([x1, y1, x2, y2], outline="blue", width=1)
    return digital_plate

# --- Main Processing Logic (Keep as before) ---
def process_file(file_path):
    if not os.path.exists(file_path): logging.error(f"File not found: {file_path}"); return []
    results_list = []
    _, file_extension = os.path.splitext(file_path); file_extension = file_extension.lower()
    filename = os.path.basename(file_path)
    logging.info(f"Processing file: {filename} (Type: {file_extension})")

    if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
        frame = cv2.imread(file_path)
        if frame is None: logging.error(f"Could not read image: {file_path}"); return []
        try: results_list.extend(process_frame(frame, 0, filename))
        except Exception as e: logging.error(f"Error processing image {filename}: {e}", exc_info=True)
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
        cap = None
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened(): logging.error(f"Could not open video: {file_path}"); return []
            frame_number = 0; frame_skip = 5
            logging.info(f"Processing video {filename}, frame skip={frame_skip}.")
            while True:
                ret, frame = cap.read()
                if not ret: logging.info(f"End of video {filename}."); break
                if frame_number % frame_skip == 0:
                    logging.debug(f"Processing frame {frame_number} of {filename}")
                    try: results_list.extend(process_frame(frame, frame_number, filename))
                    except Exception as e: logging.error(f"Error processing frame {frame_number} in {filename}: {e}", exc_info=True)
                frame_number += 1
        except Exception as video_err: logging.error(f"Error during video processing {filename}: {video_err}", exc_info=True)
        finally:
            if cap is not None and cap.isOpened(): logging.info(f"Releasing video capture {filename}"); cap.release()
    else: logging.error(f"Unsupported file type: {file_extension}"); return []
    logging.info(f"Finished {filename}. Found {len(results_list)} plates.")
    return results_list

def process_frame(frame, frame_number, filename="frame"):
    if plate_detection_model is None or char_seg_model is None:
         logging.error(f"Models missing. Cannot process frame {frame_number}."); return []
    frame_results_list = []
    logging.debug(f"Detecting plates in frame {frame_number} of {filename}")
    try: plate_results = plate_detection_model.predict(frame, verbose=False, conf=0.4)
    except Exception as e: logging.error(f"Plate detection error frame {frame_number}: {e}", exc_info=True); return []

    detected_boxes = plate_results[0].boxes
    logging.debug(f"Frame {frame_number}: Found {len(detected_boxes)} potential plates.")
    for i, plate_box in enumerate(detected_boxes):
        conf = float(plate_box.conf[0])
        x1, y1, x2, y2 = map(int, plate_box.xyxy[0].tolist())
        h_frame, w_frame = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_frame, x2), min(h_frame, y2)
        if x1 >= x2 or y1 >= y2: continue
        plate_img = frame[y1:y2, x1:x2]
        deskewed_plate = deskew_plate(plate_img)
        if deskewed_plate is None or deskewed_plate.size == 0: deskewed_plate = plate_img

        char_results = None
        try:
            if deskewed_plate is not None and deskewed_plate.shape[0] > 5 and deskewed_plate.shape[1] > 5:
                 char_results = char_seg_model.predict(deskewed_plate, verbose=False, conf=0.3)
        except Exception as e: logging.error(f"Char segmentation error frame {frame_number}, plate {i}: {e}", exc_info=True)

        ordered_characters = []
        if char_results:
            try: ordered_characters = process_and_order_characters(deskewed_plate, char_results)
            except Exception as e: logging.error(f"Char processing error frame {frame_number}, plate {i}: {e}", exc_info=True)

        digital_plate_str = None
        try:
            digital_plate_pil = create_digital_plate(deskewed_plate.shape, ordered_characters)
            digital_plate_str = to_base64(digital_plate_pil)
        except Exception as e: logging.error(f"Digital plate creation error frame {frame_number}, plate {i}: {e}", exc_info=True)

        plate_img_str = to_base64(plate_img)
        deskewed_str = to_base64(deskewed_plate)
        final_text = ''.join([c['prediction'] for c in ordered_characters]) if ordered_characters else ""
        logging.info(f"Frame {frame_number}, Plate {i}: Text='{final_text}' (Conf:{conf:.2f})")

        frame_results_list.append({
            'original_plate': plate_img_str, 'deskewed_plate': deskewed_str, 'digital_plate': digital_plate_str,
            'frame_number': frame_number, 'plate_index': i, 'confidence': conf, 'characters': ordered_characters,
            'final_text': final_text,
            'plate_dimensions': {'width': plate_img.shape[1], 'height': plate_img.shape[0]},
            'deskewed_dimensions': {'width': deskewed_plate.shape[1], 'height': deskewed_plate.shape[0]},
            'filename': filename
        })
    return frame_results_list

# --- Flask Routes (Keep as before) ---
@app.route('/', methods=['GET', 'POST'])
def upload_file_route():
    if request.method == 'POST':
        if 'file' not in request.files: flash('No file part', 'error'); return redirect(request.url)
        file = request.files['file']
        if file.filename == '': flash('No selected file', 'warning'); return redirect(request.url)
        if file:
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.mp4', '.avi', '.mov', '.mkv'}
            filename, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in allowed_extensions:
                 flash(f'Unsupported file type: {file_extension}. Allowed: {", ".join(allowed_extensions)}', 'error')
                 return redirect(request.url)
            temp_path = None; fd = None; original_filename = file.filename
            try:
                fd, temp_path = tempfile.mkstemp(suffix=file_extension, dir=app.config['UPLOAD_FOLDER'])
                logging.info(f"Created temp file: {temp_path} for {original_filename}")
                file.save(temp_path)
                if fd is not None: os.close(fd); fd = None
                results = process_file(temp_path)
                return render_template('results.html', results=results, filename=original_filename)
            except Exception as e:
                 logging.error(f"Error processing upload {original_filename}: {e}", exc_info=True)
                 flash(f'Processing error: {str(e)}', 'error')
                 return redirect(url_for('upload_file_route'))
            finally:
                 if fd is not None: # Safety close
                    try: os.close(fd)
                    except OSError: pass
                 if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path); logging.info(f"Removed temp file: {temp_path}")
                    except Exception as rm_err: logging.warning(f"Could not remove temp file {temp_path}: {rm_err}")
    return render_template('upload.html')

# --- Main Execution (Keep as before) ---
if __name__ == '__main__':
    logging.info("----- Starting ANPR Flask Application -----")
    logging.info(f"Upload Folder: {app.config['UPLOAD_FOLDER']}")
    logging.info(f"Plate Model: {PLATE_MODEL_PATH if plate_detection_model else 'Not Loaded'}")
    logging.info(f"Char Seg Model: {CHAR_SEG_MODEL_PATH if char_seg_model else 'Not Loaded'}")
    logging.info(f"Char Rec Model: {CHAR_REC_MODEL_PATH if char_recog_model else 'Not Loaded'}")
    logging.info(f"Font: {FONT_PATH if FONT_PATH else 'Default PIL Font'}")
    logging.info(f"Device: {device}")
    app.run(host='0.0.0.0', port=5001, debug=True)