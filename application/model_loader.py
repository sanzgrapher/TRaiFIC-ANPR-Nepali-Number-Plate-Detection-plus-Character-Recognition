import os
import logging
import onnxruntime as ort

import config


def _check_file(path, description):
    if not os.path.exists(path) or not os.path.isfile(path):
        logging.error(f"{description} not found at: {path}")
        return False
    logging.info(f"Found {description}: {path}")
    return True


def _onnx_path(model_path):
    """Return the expected .onnx path for a given .pt / .pth model path."""
    base, _ = os.path.splitext(model_path)
    return base + ".onnx"


def load_models():
    """Load all three inference models as ONNX Runtime sessions.

    Returns:
        plate_detection_model  – InferenceSession or None
        char_seg_model         – InferenceSession or None
        char_recog_model       – InferenceSession or None
        device                 – always "cpu" (ONNX Runtime manages execution)
        ocr_font_path          – str path to font file, or None
    """
    providers = ["CPUExecutionProvider"]

    plate_model = None
    seg_model = None
    recog_model = None

    # ------------------------------------------------------------------ #
    # Plate detection model
    # ------------------------------------------------------------------ #
    plate_onnx = _onnx_path(config.PLATE_MODEL_PATH)
    if _check_file(plate_onnx, "Plate Detection ONNX model"):
        try:
            plate_model = ort.InferenceSession(plate_onnx, providers=providers)
            logging.info("Plate Detection Model loaded successfully (ONNX).")
        except Exception as e:
            logging.error(f"Error loading Plate Detection Model: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Character segmentation model
    # ------------------------------------------------------------------ #
    seg_onnx = _onnx_path(config.CHAR_SEG_MODEL_PATH)
    if _check_file(seg_onnx, "Character Segmentation ONNX model"):
        try:
            seg_model = ort.InferenceSession(seg_onnx, providers=providers)
            logging.info("Character Segmentation Model loaded successfully (ONNX).")
        except Exception as e:
            logging.error(f"Error loading Character Segmentation Model: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Character recognition model (custom CNN)
    # ------------------------------------------------------------------ #
    recog_onnx = _onnx_path(config.CHAR_REC_MODEL_PATH)
    if _check_file(recog_onnx, "Character Recognition ONNX model"):
        try:
            recog_model = ort.InferenceSession(recog_onnx, providers=providers)
            logging.info("Character Recognition Model loaded successfully (ONNX).")
        except Exception as e:
            logging.error(f"Error loading Character Recognition Model: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # Font
    # ------------------------------------------------------------------ #
    if config.FONT_PATH and _check_file(config.FONT_PATH, "Font file"):
        ocr_font = config.FONT_PATH
    else:
        logging.warning(
            f"Font file not found or not specified: {config.FONT_PATH}. Using default PIL font."
        )
        ocr_font = None

    return plate_model, seg_model, recog_model, "cpu", ocr_font
