import os
import logging
import torch
from ultralytics import YOLO

import config
from models import NepaliPlateCNN

def check_file_exists(path, file_description):
    if not os.path.exists(path):
        logging.error(f"{file_description} not found at: {path}")
        return False
    if not os.path.isfile(path):
        logging.error(f"Path exists but is not a file: {path}")
        return False
    logging.info(f"Found {file_description}: {path}")
    return True

def load_models():
    plate_model = None
    seg_model = None
    recog_model = None
    ocr_font = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if check_file_exists(config.PLATE_MODEL_PATH, "Plate Detection Model"):
        try:
            plate_model = YOLO(config.PLATE_MODEL_PATH)
            logging.info("Plate Detection Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Plate Detection Model: {e}", exc_info=True)
            plate_model = None

    if check_file_exists(config.CHAR_SEG_MODEL_PATH, "Character Segmentation Model"):
        try:
            seg_model = YOLO(config.CHAR_SEG_MODEL_PATH)
            logging.info("Character Segmentation Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Character Segmentation Model: {e}", exc_info=True)
            seg_model = None

    if check_file_exists(config.CHAR_REC_MODEL_PATH, "Character Recognition Model"):
        try:
            recog_model = NepaliPlateCNN(num_classes=config.NUM_CLASSES)
            recog_model.load_state_dict(torch.load(config.CHAR_REC_MODEL_PATH, map_location=device))
            recog_model.to(device)
            recog_model.eval()
            logging.info("Character Recognition Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Character Recognition Model: {e}", exc_info=True)
            recog_model = None

    if config.FONT_PATH and check_file_exists(config.FONT_PATH, "Font file"):
        ocr_font = config.FONT_PATH
    else:
        logging.warning(f"Font file not found or not specified: {config.FONT_PATH}. Using default PIL font.")
        ocr_font = None

    return plate_model, seg_model, recog_model, device, ocr_font