
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
import logging

import os
import config 

def to_base64(image_pil):
    if not isinstance(image_pil, Image.Image):
        logging.warning("to_base64 received non-PIL image, attempting conversion.")
        try:
            if isinstance(image_pil, np.ndarray):
                # Assume BGR from OpenCV if it's a numpy array
                if len(image_pil.shape) == 3 and image_pil.shape[2] == 3:
                    image_pil = Image.fromarray(cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB))
                elif len(image_pil.shape) == 2: # Grayscale
                    image_pil = Image.fromarray(image_pil, mode='L')
                else:
                    raise ValueError(f"Unsupported numpy array shape: {image_pil.shape}")
            else:
                image_pil = Image.new('RGB', (50, 20), color='red')
                ImageDraw.Draw(image_pil).text((5, 5), "Error", fill="white")
        except Exception as conv_err:
            logging.error(f"Error converting input to PIL image in to_base64: {conv_err}")
            image_pil = Image.new('RGB', (50, 20), color='red')
            ImageDraw.Draw(image_pil).text((5, 5), "Error", fill="white")

    buffered = BytesIO()
    try:
        image_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        logging.error(f"Error saving image to buffer for base64 encoding: {e}")
        error_img = Image.new('RGB', (50, 20), color='red')
        ImageDraw.Draw(error_img).text((5, 5), "Encode Error", fill="white")
        error_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_str}"



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

        if maxWidth <= 0 or maxHeight <= 0:
            logging.warning(f"Invalid dimensions for perspective transform: w={maxWidth}, h={maxHeight}")
            return None

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped
    except Exception as e:
        logging.error(f"Error during four_point_transform: {e}", exc_info=True)
        return None


def preprocess_char_image(image_cv, device):
    if image_cv is None or image_cv.size == 0:
        logging.warning("preprocess_char_image received empty image.")
        return None

    if len(image_cv.shape) == 3 and image_cv.shape[2] == 3:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    elif len(image_cv.shape) == 2:
        gray = image_cv
    else:
        logging.warning(f"Unexpected image format in preprocess_char_image: shape={image_cv.shape}")
        return None
    try:
        interpolation_method = cv2.INTER_AREA if gray.shape[0] > 32 or gray.shape[1] > 32 else cv2.INTER_LINEAR
        resized = cv2.resize(gray, (32, 32), interpolation=interpolation_method)
    except cv2.error as resize_err:
         logging.error(f"OpenCV error during direct resize: {resize_err}. Image shape: {gray.shape}")
         return None
    except Exception as e:
        logging.error(f"General error during resize: {e}. Image shape: {gray.shape}")
        return None

    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor.to(device)