import cv2
import os
import logging
import time

import config
from utils import to_base64
from character_processing import deskew_plate, process_and_order_characters, create_digital_plate

def process_frame(frame, frame_number, filename_prefix, plate_model, seg_model, recog_model, device, ocr_font):
    if plate_model is None or seg_model is None or recog_model is None:
        logging.error(f"One or more models are not loaded. Cannot process frame {frame_number} from {filename_prefix}.")
        return []
    if frame is None or frame.size == 0:
         logging.error(f"Received empty frame {frame_number} for processing.")
         return []

    frame_results_list = []
    h_frame, w_frame = frame.shape[:2]
    start_time = time.time()
    logging.debug(f"Processing Frame: {frame_number} from '{filename_prefix}' ({w_frame}x{h_frame})")

    try:
        plate_results = plate_model.predict(frame, verbose=False, conf=config.PLATE_DETECT_CONF)
    except Exception as e:
        logging.error(f"Plate detection failed on frame {frame_number} ({filename_prefix}): {e}", exc_info=True)
        return [] # Cannot proceed without plate detection

    if not plate_results or not plate_results[0].boxes:
        logging.debug(f"No plates detected in frame {frame_number} ({filename_prefix}).")
        return []

    detected_boxes = plate_results[0].boxes
    logging.debug(f"Frame {frame_number}: Found {len(detected_boxes)} potential plate(s).")

    for i, plate_box in enumerate(detected_boxes):
        plate_start_time = time.time()
        plate_info = {} # Dictionary to store results for this specific plate

        try:
            conf = float(plate_box.conf[0])
            x1, y1, x2, y2 = map(int, plate_box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)

            if x1 >= x2 or y1 >= y2: # Skip invalid boxes
                logging.warning(f"Skipping invalid plate box {i} in frame {frame_number}: [{x1},{y1},{x2},{y2}]")
                continue

            plate_img = frame[y1:y2, x1:x2].copy() # Extract with copy
            plate_info['original_plate'] = to_base64(plate_img)
            plate_info['plate_dimensions'] = {'width': plate_img.shape[1], 'height': plate_img.shape[0]}
            plate_info['confidence'] = conf
            plate_info['frame_number'] = frame_number
            plate_info['plate_index'] = i
            plate_info['filename'] = filename_prefix

            deskewed_plate = deskew_plate(plate_img)
            if deskewed_plate is None or deskewed_plate.size == 0:
                logging.warning(f"Deskewing returned empty image for plate {i}, frame {frame_number}. Using original.")
                deskewed_plate = plate_img # Fallback to original crop
            plate_info['deskewed_plate'] = to_base64(deskewed_plate)
            plate_info['deskewed_dimensions'] = {'width': deskewed_plate.shape[1], 'height': deskewed_plate.shape[0]}


            char_seg_results = None
            ordered_characters = []
            final_text = ""
            digital_plate_str = to_base64(None) # Placeholder

            if deskewed_plate is not None and deskewed_plate.shape[0] > 5 and deskewed_plate.shape[1] > 5:
                 try:
                     char_seg_results = seg_model.predict(deskewed_plate, verbose=False, conf=config.CHAR_SEG_CONF)
                 except Exception as e:
                     logging.error(f"Char segmentation failed for plate {i}, frame {frame_number}: {e}", exc_info=True)

                 if char_seg_results:
                     try:
                         ordered_characters, final_text = process_and_order_characters(
                             deskewed_plate, char_seg_results, recog_model, device
                         )
                         plate_info['characters'] = ordered_characters
                         plate_info['final_text'] = final_text
                         logging.info(f"Frame {frame_number}, Plate {i}: Recognized Text='{final_text}' (Plate Conf:{conf:.2f})")
                     except Exception as e:
                         logging.error(f"Character processing/ordering failed for plate {i}, frame {frame_number}: {e}", exc_info=True)
                         plate_info['characters'] = []
                         plate_info['final_text'] = "[OCR Error]"


            try:
                digital_plate_pil = create_digital_plate(deskewed_plate.shape[:2], plate_info.get('characters', []), ocr_font)
                digital_plate_str = to_base64(digital_plate_pil)
            except Exception as e:
                logging.error(f"Digital plate creation failed for plate {i}, frame {frame_number}: {e}", exc_info=True)

            plate_info['digital_plate'] = digital_plate_str

            frame_results_list.append(plate_info)
            plate_end_time = time.time()
            logging.debug(f"Plate {i} processed in {plate_end_time - plate_start_time:.3f} seconds.")

        except Exception as plate_proc_err:
            logging.error(f"Error processing detected plate {i} in frame {frame_number} ({filename_prefix}): {plate_proc_err}", exc_info=True)
            frame_results_list.append({
                'error': f"Failed to process plate {i}",
                'frame_number': frame_number,
                'plate_index': i,
                'filename': filename_prefix,
                'original_plate': to_base64(plate_img) if 'plate_img' in locals() else to_base64(None), # Add original if possible
            })
            continue # Move to the next detected plate

    end_time = time.time()
    logging.debug(f"Frame {frame_number} processing took {end_time - start_time:.3f} seconds. Found {len(frame_results_list)} valid plates.")
    return frame_results_list


def process_file(file_path, plate_model, seg_model, recog_model, device, ocr_font):

    if not os.path.exists(file_path):
        logging.error(f"Input file not found: {file_path}")
        return []

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    filename = os.path.basename(file_path)
    results_list = []
    total_start_time = time.time()
    logging.info(f"Starting processing for file: {filename} (Type: {file_extension})")

    if file_extension in config.ALLOWED_EXTENSIONS:
        if file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']:
            try:
                frame = cv2.imread(file_path)
                if frame is None:
                    logging.error(f"Could not read image file: {file_path}")
                    return []
                results_list = process_frame(frame, 0, filename, plate_model, seg_model, recog_model, device, ocr_font)
            except Exception as e:
                logging.error(f"Error processing image {filename}: {e}", exc_info=True)

        elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
            cap = None
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    logging.error(f"Could not open video file: {file_path}")
                    return []

                frame_number = 0
                processed_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logging.info(f"End of video {filename}.")
                        break

                    if frame_number % config.VIDEO_FRAME_SKIP == 0:
                        logging.debug(f"Processing video frame {frame_number} of {filename}")
                        try:
                            frame_results = process_frame(frame, frame_number, filename, plate_model, seg_model, recog_model, device, ocr_font)
                            if frame_results: # Only add if plates were found in this frame
                                results_list.extend(frame_results)
                            processed_count += 1
                        except Exception as e:
                            logging.error(f"Error processing frame {frame_number} in video {filename}: {e}", exc_info=True)

                    frame_number += 1


            except Exception as video_err:
                logging.error(f"Critical error during video processing {filename}: {video_err}", exc_info=True)
            finally:
                if cap is not None and cap.isOpened():
                    logging.debug(f"Releasing video capture for {filename}")
                    cap.release()
        else:
             logging.error(f"File type {file_extension} seems allowed but has no processing logic.")

    else:
        logging.error(f"Unsupported file type received by process_file: {file_extension}")


    total_end_time = time.time()
    logging.info(f"Finished processing {filename} in {total_end_time - total_start_time:.3f} seconds. Found {len(results_list)} plates total.")
    return results_list