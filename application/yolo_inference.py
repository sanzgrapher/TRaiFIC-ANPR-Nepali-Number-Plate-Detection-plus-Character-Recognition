"""
YOLO ONNX inference helpers.

Provides letterboxing, NMS, and an ultralytics-compatible result adapter so
that the rest of the application can call yolo_predict() and access results
via the same .boxes / .xyxy / .conf interface that was used with ultralytics.
"""

import cv2
import logging
import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def letterbox(img_bgr, new_shape=(640, 640)):
    """Resize *img_bgr* to *new_shape* with letterbox padding (grey borders).

    Returns:
        img_padded  – padded image, shape (*new_shape*, 3), dtype uint8
        scale       – the single scale factor applied to both dimensions
        pad         – (pad_left, pad_top) pixels of padding added
    """
    h, w = img_bgr.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)
    target_w = int(round(w * scale))
    target_h = int(round(h * scale))

    img_resized = cv2.resize(img_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    pad_left = (new_w - target_w) // 2
    pad_right = new_w - target_w - pad_left
    pad_top = (new_h - target_h) // 2
    pad_bottom = new_h - target_h - pad_top

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return img_padded, scale, (pad_left, pad_top)


# ---------------------------------------------------------------------------
# NMS (pure NumPy — no scipy/torch needed)
# ---------------------------------------------------------------------------

def _nms(boxes, scores, iou_threshold=0.45):
    """Return indices of boxes to keep after non-maximum suppression.

    Args:
        boxes: float32 array [N, 4] in x1,y1,x2,y2 format
        scores: float32 array [N]
        iou_threshold: float

    Returns:
        list of int indices
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


# ---------------------------------------------------------------------------
# Ultralytics-compatible result adapters
# ---------------------------------------------------------------------------

class _Box:
    """Minimal adapter mimicking a single ultralytics bounding-box entry."""

    def __init__(self, x1, y1, x2, y2, conf):
        # xyxy stored as 2-D array so that box.xyxy[0] returns a 1-D array
        self._xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
        self._conf = np.array([conf], dtype=np.float32)

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def conf(self):
        return self._conf


class _Boxes:
    """Minimal adapter mimicking ultralytics Boxes."""

    def __init__(self, box_list):
        self._list = box_list

    def __bool__(self):
        return len(self._list) > 0

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)


class _Result:
    """Minimal adapter mimicking one element of the ultralytics Results list."""

    def __init__(self, box_list):
        self.boxes = _Boxes(box_list)


# ---------------------------------------------------------------------------
# Main prediction entry-point
# ---------------------------------------------------------------------------

def yolo_predict(session, img_bgr, conf_threshold=0.4):
    """Run a YOLO ONNX model and return ultralytics-compatible results.

    The function mirrors the interface of ``ultralytics.YOLO.predict()``:
    it returns a list whose first element has a ``.boxes`` attribute
    containing individual ``_Box`` objects with ``.xyxy`` and ``.conf``.

    The ONNX model must have been exported from Ultralytics YOLOv8/v10/v11/v12
    with the default ``model.export(format='onnx')`` command, which produces:
      - Input  ``images``:  [1, 3, 640, 640], float32, range [0, 1]
      - Output ``output0``: [1, 4+num_classes, num_anchors], float32
        where the first 4 rows are cx, cy, w, h in 640×640 pixel space.

    Args:
        session:        onnxruntime.InferenceSession
        img_bgr:        original frame/image as a BGR uint8 numpy array
        conf_threshold: minimum detection confidence

    Returns:
        list of _Result (always length 1, matching ultralytics behaviour)
    """
    if img_bgr is None or img_bgr.size == 0:
        logging.error("yolo_predict received an empty image.")
        return [_Result([])]

    h_orig, w_orig = img_bgr.shape[:2]

    # --- Preprocess ---
    img_lb, scale, (pad_left, pad_top) = letterbox(img_bgr, (640, 640))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_input = img_rgb.astype(np.float32) / 255.0
    img_input = img_input.transpose(2, 0, 1)[np.newaxis]       # [1, 3, 640, 640]
    img_input = np.ascontiguousarray(img_input)

    # --- Inference ---
    try:
        input_name = session.get_inputs()[0].name
        raw = session.run(None, {input_name: img_input})[0]    # [1, 4+nc, N]
    except Exception as e:
        logging.error(f"ONNX inference failed: {e}", exc_info=True)
        return [_Result([])]

    # --- Decode ---
    preds = raw[0].T                                            # [N, 4+nc]
    box_preds = preds[:, :4]                                    # cx,cy,w,h in 640-px space
    cls_preds = preds[:, 4:]
    conf = cls_preds.max(axis=1)

    mask = conf > conf_threshold
    if not mask.any():
        return [_Result([])]

    box_preds = box_preds[mask]
    conf = conf[mask]

    # cx,cy,w,h -> x1,y1,x2,y2 (still 640-px space)
    x1_640 = box_preds[:, 0] - box_preds[:, 2] / 2
    y1_640 = box_preds[:, 1] - box_preds[:, 3] / 2
    x2_640 = box_preds[:, 0] + box_preds[:, 2] / 2
    y2_640 = box_preds[:, 1] + box_preds[:, 3] / 2

    # Undo letterbox → original image coordinates
    x1 = np.clip((x1_640 - pad_left) / scale, 0, w_orig)
    y1 = np.clip((y1_640 - pad_top)  / scale, 0, h_orig)
    x2 = np.clip((x2_640 - pad_left) / scale, 0, w_orig)
    y2 = np.clip((y2_640 - pad_top)  / scale, 0, h_orig)

    boxes_arr = np.stack([x1, y1, x2, y2], axis=1)
    keep = _nms(boxes_arr, conf, iou_threshold=0.45)

    box_list = [_Box(x1[i], y1[i], x2[i], y2[i], conf[i]) for i in keep]
    logging.debug(f"yolo_predict: {len(box_list)} boxes after NMS (conf>{conf_threshold})")
    return [_Result(box_list)]
