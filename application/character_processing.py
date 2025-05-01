import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import logging
import os

import config
from utils import preprocess_char_image, to_base64, order_points, four_point_transform

def deskew_plate(plate_img):
  
    if plate_img is None or plate_img.size == 0:
        logging.warning("Deskew received an empty image.")
        return plate_img

    h_orig, w_orig = plate_img.shape[:2]
    if h_orig < config.DESKEW_MIN_PLATE_HEIGHT or w_orig < config.DESKEW_MIN_PLATE_WIDTH:
        logging.debug(f"Plate too small for deskewing ({w_orig}x{h_orig}). Skipping.")
        return plate_img

    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logging.debug("No contours found for deskewing.")
            return plate_img

        possible_plates = []
        plate_area_orig = h_orig * w_orig
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0.05 * plate_area_orig and area < 0.95 * plate_area_orig:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    possible_plates.append(approx)

        if possible_plates:
            largest_quad = max(possible_plates, key=cv2.contourArea)
            screenCnt = largest_quad.reshape(4, 2)
            warped = four_point_transform(plate_img, screenCnt.astype("float32"))

            if warped is not None and warped.shape[0] >= config.DESKEW_MIN_PLATE_HEIGHT // 2 and warped.shape[1] >= config.DESKEW_MIN_PLATE_WIDTH // 2:
                 logging.debug("Deskew successful using approxPolyDP.")
                 return warped
            else:
                 logging.debug("Warped image from approxPolyDP is invalid or too small.")

        logging.debug("Falling back to minAreaRect for deskewing.")
        largest_contour = max(contours, key=cv2.contourArea) # Use largest overall contour
        rect = cv2.minAreaRect(largest_contour) # (center(x,y), (width, height), angle)
        box = cv2.boxPoints(rect) # Get the 4 corners of the rotated rectangle
        box = np.intp(box) # Convert to integers

        rect_width, rect_height = rect[1]
        if rect_width <= 0 or rect_height <=0: return plate_img # Avoid division by zero
        aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)

        if rect_width > config.DESKEW_MIN_PLATE_WIDTH / 2 and rect_height > config.DESKEW_MIN_PLATE_HEIGHT / 2 and aspect_ratio < 10:
             warped = four_point_transform(plate_img, box.astype("float32"))
             if warped is not None and warped.shape[0] >= config.DESKEW_MIN_PLATE_HEIGHT // 2 and warped.shape[1] >= config.DESKEW_MIN_PLATE_WIDTH // 2:
                 logging.debug("Deskew successful using minAreaRect.")
                 return warped
             else:
                 logging.debug("Warped image from minAreaRect is invalid or too small.")

        logging.debug("Deskewing failed to find a suitable transform. Returning original.")
        return plate_img # Return original if all attempts fail

    except Exception as e:
        logging.error(f"Error during deskewing: {e}", exc_info=True)
        return plate_img # Return original on any error

def process_and_order_characters(deskewed_plate, char_seg_results, char_recog_model, device):

    if char_recog_model is None:
        logging.error("Character recognition model not loaded. Cannot process characters.")
        return [], ""
    if deskewed_plate is None or deskewed_plate.size == 0:
        logging.warning("Cannot process characters from empty plate image.")
        return [], ""

    characters_data = [] 
    h_plate, w_plate = deskewed_plate.shape[:2]

    if not char_seg_results or not hasattr(char_seg_results[0], 'boxes') or not char_seg_results[0].boxes:
        logging.debug("No character bounding boxes found in segmentation results.")
        return [], ""

    for box in char_seg_results[0].boxes:
        seg_conf = float(box.conf[0]) if box.conf is not None else 1.0
        if seg_conf < config.CHAR_SEG_CONF: # Use confidence from config
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_plate, x2), min(h_plate, y2)

        if x1 >= x2 or y1 >= y2:
            continue

        char_img_cv = deskewed_plate[y1:y2, x1:x2]

        input_tensor = preprocess_char_image(char_img_cv, device)
        if input_tensor is None:
            logging.warning(f"Preprocessing failed for character at [{x1},{y1},{x2},{y2}]")
            continue

        try:
            with torch.no_grad(): # Inference mode
                output = char_recog_model(input_tensor)
                probabilities = F.softmax(output, dim=1)
                rec_conf, pred_idx = torch.max(probabilities, 1)

                pred_label = "?"
                if pred_idx.item() < len(config.CLASS_LABELS):
                    pred_label = config.CLASS_LABELS[pred_idx.item()]
                else:
                    logging.error(f"Prediction index {pred_idx.item()} out of range for CLASS_LABELS (size {len(config.CLASS_LABELS)}).")

                rec_confidence = float(rec_conf.item())

                characters_data.append({
                    'prediction': pred_label,
                    'confidence': rec_confidence,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cx': (x1 + x2) / 2.0, # Center X
                    'cy': (y1 + y2) / 2.0, # Center Y
                    'height': y2 - y1,
                    'char_img_cv': char_img_cv # Keep CV image for potential base64 later
                })
                logging.debug(f"Char @ [{x1},{y1}-{x2},{y2}]: '{pred_label}' (Conf: {rec_confidence:.2f})")

        except Exception as rec_err:
            logging.error(f"Error during character recognition step for char at [{x1},{y1},{x2},{y2}]: {rec_err}", exc_info=True)
            continue

    if not characters_data:
        logging.debug("No characters recognized after processing boxes.")
        return [], ""

    characters_data.sort(key=lambda c: (c['cy'], c['cx']))

    lines = []
    if characters_data:
        current_line = [characters_data[0]]
        valid_heights = [c['height'] for c in characters_data if c['height'] > 0]
        avg_char_height = np.mean(valid_heights) if valid_heights else h_plate / 4 
        if avg_char_height <= 0 : avg_char_height = 10 

        line_y_threshold = avg_char_height * config.CHAR_ORDERING_HEIGHT_FRACTION

        for i in range(1, len(characters_data)):
            prev_char = current_line[-1]
            curr_char = characters_data[i]

            y_diff = abs(curr_char['cy'] - prev_char['cy'])

            if y_diff < line_y_threshold:
                current_line.append(curr_char)
            else:
                lines.append(current_line)
                current_line = [curr_char]

        lines.append(current_line) 

    ordered_char_list_final = []
    for line in lines:
        line.sort(key=lambda c: c['cx'])
        ordered_char_list_final.extend(line) 

    final_char_output_list = []
    line1_text = ""
    line2_text = ""

    line_split_y = h_plate 
    if len(lines) > 1:
        try:
            last_y_line1 = lines[0][-1]['cy']
            first_y_line2 = lines[1][0]['cy']
            line_split_y = (last_y_line1 + first_y_line2) / 2.0
            logging.info(f"Detected {len(lines)} lines. Split Y estimated at: {line_split_y:.1f}")
        except IndexError:
             logging.warning("Error calculating line split Y, treating as single line.")
             line_split_y = h_plate # Fallback


    for char_data in ordered_char_list_final:
        img_str = None
        try:
            char_pil = Image.fromarray(cv2.cvtColor(char_data['char_img_cv'], cv2.COLOR_BGR2RGB))
            img_str = to_base64(char_pil)
        except Exception as img_conv_err:
            logging.error(f"Error converting char img to base64: {img_conv_err}")
            img_str = to_base64(None) # Get placeholder error image

        final_char_output_list.append({
            'prediction': char_data['prediction'],
            'confidence': char_data['confidence'],
            'x1': char_data['x1'], 'y1': char_data['y1'],
            'x2': char_data['x2'], 'y2': char_data['y2'],
            'image': img_str
        })

        if char_data['confidence'] >= config.CHAR_REC_CONF_THRESHOLD:
            if char_data['cy'] < line_split_y:
                line1_text += char_data['prediction']
            else:
                line2_text += char_data['prediction']

    final_text = line1_text
    if line2_text: # Add space only if line 2 has text
        final_text += " " + line2_text

    logging.info(f"Ordered characters. Line 1: '{line1_text}', Line 2: '{line2_text}'. Final: '{final_text}'")
    return final_char_output_list, final_text.strip()


# --- Digital Plate Creation ---
def create_digital_plate(plate_shape, characters, font_path):

    h, w = plate_shape[0], plate_shape[1]
    if w <= 0 or h <= 0:
        logging.warning(f"Invalid plate shape for digital plate: {plate_shape}")
        digital_plate = Image.new('RGB', (150, 50), (200, 200, 200)) # Gray background
        draw = ImageDraw.Draw(digital_plate)
        draw.text((5, 5), "Invalid Size", fill="red", font=ImageFont.load_default())
        return digital_plate

    digital_plate = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(digital_plate)

    font_size = 30 # Default size
    if characters:
        valid_heights = [c['y2'] - c['y1'] for c in characters if (c['y2'] - c['y1']) > 0]
        if valid_heights:
            avg_char_h = np.mean(valid_heights)
            font_size = max(10, min(int(h * 0.8), int(avg_char_h * 0.9)))
        else:
            font_size = max(10, int(h * 0.6)) # Fallback if no valid heights
    else:
        font_size = max(10, int(h * 0.6)) # Fallback if no characters

    font = None
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            logging.warning(f"Font not found at {font_path}, using default.")
            try:
                font = ImageFont.load_default()
                font_size = 10 # Default font is usually small
            except IOError:
                 logging.error("Could not load default PIL font.")
                 font = None # No font available
    except Exception as font_err:
        logging.error(f"Error loading font '{font_path}': {font_err}. Using default.")
        try: font = ImageFont.load_default(); font_size = 10
        except IOError: font = None


    if font:
        logging.debug(f"Using font: {font_path or 'Default'} with size {font_size}")
        for char in characters:
             if char['confidence'] >= config.CHAR_REC_CONF_THRESHOLD:
                text = char['prediction']
                x1, y1, x2, y2 = char['x1'], char['y1'], char['x2'], char['y2']
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0

                try:
                    if hasattr(draw, 'textbbox'):
                         text_width, text_height = draw.textlength(text, font=font), font_size 
                         draw_x = x_center - text_width / 2
                         draw_y = y_center - text_height / 2 # Rough vertical center
                         draw.text((draw_x, draw_y), text, font=font, fill=(0, 0, 0))
                    else: # Older PIL version fallback
                        text_width, text_height = draw.textsize(text, font=font)
                        draw_x = x_center - text_width / 2
                        draw_y = y_center - text_height / 2
                        draw.text((draw_x, draw_y), text, font=font, fill=(0, 0, 0))

                except Exception as draw_err:
                    logging.error(f"Error drawing text '{text}' at ({x_center:.1f}, {y_center:.1f}): {draw_err}")
                    draw.rectangle([x1, y1, x2, y2], outline="blue", width=1)
    else:
        logging.error("Cannot draw text on digital plate: No font loaded.")
        for char in characters:
            if char['confidence'] >= config.CHAR_REC_CONF_THRESHOLD:
                 draw.rectangle([char['x1'], char['y1'], char['x2'], char['y2']], outline="red", width=1)

    return digital_plate