# inference.py
"""
Hierarchical Helmet & Seatbelt Inference with ANPR + OCR
- Resize large images
- High-res vehicle detection fallback (imgsz=1280, low conf)
- Center-crop vehicle detection fallback
- Hierarchical logic: vehicle -> persons -> helmet/seatbelt
- ANPR detection + OCR (EasyOCR primary, pytesseract fallback)
- Saves annotated images and plate crops for violations
"""

import os
import sys
import cv2
import re
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

# ultralytics YOLO
from ultralytics import YOLO

# OCR libs (try EasyOCR first; fallback to pytesseract)
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    _PYTESSERACT_AVAILABLE = True
except Exception:
    _PYTESSERACT_AVAILABLE = False

# -----------------------
# Logger
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("HSDS-Inference")

# -----------------------
# Config paths (edit if needed)
# -----------------------
MODEL_DIR = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\models")
OUTPUT_DIR = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\static\uploads\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default test image (override via CLI)
DEFAULT_TEST_IMAGE = Path(
    r"P:\Helmet_Seatbelt_Detection_system\backend\app\static\uploads\63f6c10cb87a457b9b59dcbd246a7962_test_img.jpg"
)

# -----------------------
# Thresholds & parameters
# -----------------------
DEFAULT_DETECT_CONF = 0.10
VEHICLE_FALLBACK_CONF = 0.03
HELMET_CONF_THRESHOLD = 0.25
SEATBELT_CONF_THRESHOLD = 0.25
IOU_PERSON_HELMET = 0.15
IOU_PERSON_SEATBELT = 0.10
PLATE_IOU_THRESHOLD = 0.10

# When images are very large (> MAX_DIM), resize them first to keep objects
MAX_DIM = 1600  # max width/height after resizing (keeps aspect ratio)

# Center-crop fraction for vehicle pass (helpful when object roughly centered)
CENTER_CROP = {"x0": 0.12, "x1": 0.88, "y0": 0.28, "y1": 0.96}

# -----------------------
# Utility helpers
# -----------------------
def _choose_device(device_pref="auto"):
    import torch
    if device_pref == "cpu":
        return "cpu"
    if device_pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"

def _normalize_bbox_global(val):
    """Normalize ultralytics box structures to [x1,y1,x2,y2] floats or None"""
    if val is None:
        return None
    # If it's already a list
    if isinstance(val, list):
        vals = val
    else:
        try:
            vals = val.tolist()
        except Exception:
            try:
                vals = list(val)
            except Exception:
                return None
    # nested list => use first
    if len(vals) > 0 and isinstance(vals[0], (list, tuple)):
        vals = vals[0]
    flat = []
    for v in vals:
        if isinstance(v, (list, tuple)):
            flat.extend(v)
        else:
            flat.append(v)
    if len(flat) < 4:
        return None
    try:
        return [float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])]
    except Exception:
        return None

def expand_bbox(bbox, frame_shape, pct=0.25):
    """Expand bbox by pct (fraction) in all directions, clamp to frame."""
    if not bbox or len(bbox) < 4:
        return bbox
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * pct
    pad_y = bh * pct
    nx1 = max(0, int(x1 - pad_x))
    ny1 = max(0, int(y1 - pad_y))
    nx2 = min(w - 1, int(x2 + pad_x))
    ny2 = min(h - 1, int(y2 + pad_y))
    return [nx1, ny1, nx2, ny2]

def _iou(boxA, boxB):
    if not boxA or not boxB:
        return 0.0
    try:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
        denom = boxAArea + boxBArea - interArea
        if denom <= 0:
            return 0.0
        return interArea / denom
    except Exception:
        return 0.0

def _clean_plate_text(s):
    if not s:
        return None
    s = str(s).upper().strip()
    s = re.sub(r'[^A-Z0-9]', '', s)
    if len(s) < 3:
        return None
    return s

def _save_crop_image(image, bbox, prefix="crop"):
    if image is None or bbox is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2].copy()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fn = OUTPUT_DIR / f"{prefix}_license_plate_crop_{ts}.jpg"
    cv2.imwrite(str(fn), crop)
    return str(fn)

def _save_annotated_image(image, prefix="annotated"):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fn = OUTPUT_DIR / f"{prefix}_{ts}.jpg"
    cv2.imwrite(str(fn), image)
    return str(fn)

# -----------------------
# OCR utilities
# -----------------------
_ez_reader = None

def _init_easyocr(lang_list=["en"]):
    global _ez_reader
    if _ez_reader is None and _EASYOCR_AVAILABLE:
        try:
            _ez_reader = easyocr.Reader(lang_list, gpu=False)
        except Exception as e:
            logger.warning("EasyOCR init failed: %s", e)
            _ez_reader = None
    return _ez_reader

def extract_plate_text_from_crop(crop_bgr):
    """Try EasyOCR first, then pytesseract fallback. Return (text, confidence)"""
    if crop_bgr is None:
        return None, 0.0
    # Preprocess: convert to grayscale, increase contrast
    try:
        img = crop_bgr.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # apply histogram equalization and slight sharpening
        gray = cv2.equalizeHist(gray)
        # try a few scales
        h, w = gray.shape[:2]
        scale = 1.0
        if max(h, w) < 200:
            scale = 2.0
            gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)
        # threshold adaptively might help
        # try EasyOCR
        if _EASYOCR_AVAILABLE:
            _init_easyocr()
            try:
                res = _ez_reader.readtext(gray)
                if res:
                    # choose longest cleaned text with a score
                    best = None
                    for (bbox, txt, conf) in res:
                        cleaned = _clean_plate_text(txt)
                        if cleaned:
                            if best is None or (len(cleaned) > len(best[0])) or (conf > best[1]):
                                best = (cleaned, float(conf))
                    if best:
                        return best[0], float(best[1])
            except Exception as e:
                logger.debug("EasyOCR error: %s", e)

        # Fallback to pytesseract
        if _PYTESSERACT_AVAILABLE:
            try:
                cfg = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                txt = pytesseract.image_to_string(gray, config=cfg)
                cleaned = _clean_plate_text(txt)
                if cleaned:
                    # pytesseract doesn't return confidence easily here, so set small default
                    return cleaned, 0.3
            except Exception as e:
                logger.debug("pytesseract error: %s", e)

    except Exception as e:
        logger.exception("OCR preproc failed: %s", e)

    return None, 0.0

# -----------------------
# Load models safely
# -----------------------
def safe_load_model(path: Path, device_choice="auto"):
    if not path.exists():
        logger.error("Model file not found: %s", path)
        return None
    try:
        model = YOLO(str(path))
        device = _choose_device(device_choice)
        model.to(device)
        logger.info("Loaded model %s on %s", path.name, device.upper())
        return model
    except Exception as e:
        logger.exception("Failed to load model %s: %s", path, e)
        return None

# -----------------------
# Load all models (lazy)
# -----------------------
models = {
    "vehicle": safe_load_model(MODEL_DIR / "vehicle_classifier.pt"),
    "person_head": safe_load_model(MODEL_DIR / "Person_and_head_detector.pt"),
    "helmet": safe_load_model(MODEL_DIR / "helmet_detector.pt"),
    "seatbelt": safe_load_model(MODEL_DIR / "seatbelt_detector.pt"),
    "anpr": safe_load_model(MODEL_DIR / "anpr_detector.pt"),
}

# -----------------------
# Run models on frame: wrapper producing normalized outputs
# -----------------------
def run_all_models_on_frame(frame, imgsz=640, conf=DEFAULT_DETECT_CONF):
    outputs = {}
    for name, model in models.items():
        outputs[name] = []
        if model is None:
            continue
        try:
            results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
            dets = []
            for r in results:
                for box in getattr(r, "boxes", []):
                    try:
                        cls_id = int(getattr(box, "cls", 0))
                    except Exception:
                        cls_id = 0
                    label = None
                    try:
                        label = model.model.names[cls_id] if hasattr(model, "model") and hasattr(model.model, "names") else str(cls_id)
                    except Exception:
                        label = str(cls_id)
                    confv = float(getattr(box, "conf", 0.0) or 0.0)
                    xyxy = None
                    if hasattr(box, "xyxy"):
                        xyxy = _normalize_bbox_global(getattr(box, "xyxy", None))
                    elif hasattr(box, "xywh"):
                        try:
                            vals = box.xywh.tolist()
                            if len(vals) >= 4:
                                x, y, w, h = vals[:4]
                                xyxy = [float(x - w/2), float(y - h/2), float(x + w/2), float(y + h/2)]
                        except Exception:
                            xyxy = _normalize_bbox_global(getattr(box, "xywh", None))
                    else:
                        try:
                            xyxy = _normalize_bbox_global(box.tolist())
                        except Exception:
                            xyxy = None
                    dets.append({"label": label, "confidence": round(confv, 4), "bbox": xyxy})
            outputs[name] = dets
        except Exception as e:
            logger.exception("Error running model %s: %s", name, e)
            outputs[name] = []
    return outputs

# -----------------------
# Vehicle detection fallback pass
# -----------------------
def vehicle_fallback_pass(frame):
    """Try higher-resolution & center-crop passes for vehicle detection."""
    vehicle_model = models.get("vehicle")
    if vehicle_model is None:
        return []

    try:
        # high-res pass
        try:
            results = vehicle_model.predict(source=frame, conf=VEHICLE_FALLBACK_CONF, imgsz=1280, verbose=False)
            dets = []
            for r in results:
                for box in getattr(r, "boxes", []):
                    cls_id = int(getattr(box, "cls", 0))
                    label = vehicle_model.model.names.get(cls_id) if hasattr(vehicle_model, "model") and hasattr(vehicle_model.model, "names") else str(cls_id)
                    confv = float(getattr(box, "conf", 0.0) or 0.0)
                    xyxy = None
                    if hasattr(box, "xyxy"):
                        xyxy = _normalize_bbox_global(getattr(box, "xyxy", None))
                    dets.append({"label": label, "confidence": round(confv, 4), "bbox": xyxy})
            if dets:
                logger.info("Vehicle fallback: high-res pass found %d vehicle(s)", len(dets))
                return dets
        except Exception:
            logger.exception("High-res vehicle pass failed")

        # center-crop pass
        h, w = frame.shape[:2]
        cx0 = int(w * CENTER_CROP["x0"])
        cy0 = int(h * CENTER_CROP["y0"])
        cx1 = int(w * CENTER_CROP["x1"])
        cy1 = int(h * CENTER_CROP["y1"])
        crop = frame[cy0:cy1, cx0:cx1].copy()
        if crop.size > 0:
            try:
                results = vehicle_model.predict(source=crop, conf=VEHICLE_FALLBACK_CONF, imgsz=1280, verbose=False)
                dets = []
                for r in results:
                    for box in getattr(r, "boxes", []):
                        cls_id = int(getattr(box, "cls", 0))
                        label = vehicle_model.model.names.get(cls_id) if hasattr(vehicle_model, "model") and hasattr(vehicle_model.model, "names") else str(cls_id)
                        confv = float(getattr(box, "conf", 0.0) or 0.0)
                        xyxy = None
                        if hasattr(box, "xyxy"):
                            raw = _normalize_bbox_global(getattr(box, "xyxy", None))
                            # convert crop coords back to full-frame coords
                            if raw:
                                x1, y1, x2, y2 = raw
                                x1 += cx0; x2 += cx0
                                y1 += cy0; y2 += cy0
                                xyxy = [x1, y1, x2, y2]
                        dets.append({"label": label, "confidence": round(confv, 4), "bbox": xyxy})
                if dets:
                    logger.info("Vehicle fallback: center-crop pass found %d vehicle(s)", len(dets))
                    return dets
            except Exception:
                logger.exception("Center-crop vehicle pass failed")

    except Exception:
        logger.exception("vehicle_fallback_pass encountered error")

    return []

# -----------------------
# Drawing helpers
# -----------------------
BOX_COLOR_VIOLATION = (0, 0, 255)
BOX_COLOR_OK = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def draw_bbox_and_label(frame, bbox, label_text, is_violation=False):
    if not bbox or len(bbox) < 4:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w-1))
    y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h-1))
    color = BOX_COLOR_VIOLATION if is_violation else BOX_COLOR_OK
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = label_text
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    tx = x1
    ty = max(10, y1 - 8)
    cv2.rectangle(frame, (tx, ty - th - 6), (min(w-1, tx + tw + 6), ty + 4), TEXT_BG, -1)
    cv2.putText(frame, text, (tx, ty), FONT, 0.6, TEXT_COLOR, 2)

# -----------------------
# Main detection logic
# -----------------------
def detect_violations(frame, source_path=None, device_pref="auto", debug=False):
    """
    Hierarchical detection that returns a list of summaries:
    Each summary contains:
      - vehicle_type ("Twowheeler" / "MoreThanTwoWheeler" / None)
      - violation ("No Helmet" / "No Seatbelt" / None)
      - number_plate_model_label
      - number_plate_model_confidence
      - number_plate_text (OCR)
      - number_plate_text_confidence
      - saved_crop_path, saved_annotated_path
      - metadata (per-model confidences)
    """
    start_ts = time.time()

    # Resize large images to keep objects visible
    h0, w0 = frame.shape[:2]
    if max(h0, w0) > MAX_DIM:
        scale = MAX_DIM / max(h0, w0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        logger.info("Resizing frame %dx%d -> %dx%d to preserve small objects", w0, h0, nw, nh)
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    # 1) Run default pass (all models at imgsz=640)
    model_outputs = run_all_models_on_frame(frame, imgsz=640, conf=DEFAULT_DETECT_CONF)
    if debug:
        logger.info("DEBUG: raw model outputs (truncated):")
        for k, v in model_outputs.items():
            logger.info("  %s -> %s", k, v)

    # 2) If vehicle list empty, run fallback passes
    if not model_outputs.get("vehicle"):
        extra = vehicle_fallback_pass(frame)
        if extra:
            model_outputs["vehicle"] = extra

    # convenience lists
    vehicles = model_outputs.get("vehicle") or []
    persons = model_outputs.get("person_head") or []
    helmets = model_outputs.get("helmet") or []
    seatbelts = model_outputs.get("seatbelt") or []
    anpr = model_outputs.get("anpr") or []

    # quick global confidences
    helmet_conf_global = max((h.get("confidence") or 0) for h in helmets) if helmets else 0.0
    seatbelt_conf_global = max((s.get("confidence") or 0) for s in seatbelts) if seatbelts else 0.0
    anpr_conf_global = max((a.get("confidence") or 0) for a in anpr) if anpr else 0.0

    summaries = []

    # Helper to pick plate overlapping
    def _pick_plate_for_bbox(vbbox):
        if not anpr:
            return None, 0.0, None
        for a in anpr:
            ab = a.get("bbox")
            if vbbox and ab and _iou(vbbox, ab) > PLATE_IOU_THRESHOLD:
                return a.get("label"), a.get("confidence", 0.0), ab
        # fallback to best anpr
        return anpr[0].get("label"), anpr[0].get("confidence", 0.0), anpr[0].get("bbox")

    # If vehicles detected -> handle each vehicle
    if vehicles:
        for v in vehicles:
            v_label_raw = v.get("label")
            v_conf = float(v.get("confidence") or 0.0)
            v_bbox = v.get("bbox")
            # Map vehicle label to Twowheeler / MoreThanTwoWheeler
            v_label_norm = None
            if v_label_raw:
                lr = str(v_label_raw).lower()
                if "two" in lr or "motor" in lr or "bike" in lr or "scooter" in lr:
                    v_label_norm = "Twowheeler"
                else:
                    v_label_norm = "MoreThanTwoWheeler"

            # find overlapping persons
            linked_persons = []
            for p in persons:
                pb = p.get("bbox")
                if v_bbox and pb and _iou(v_bbox, pb) > 0.05:
                    linked_persons.append(p)

            violation_types = []
            saved_crop = None
            saved_annotated = None
            plate_model_label, plate_model_conf, plate_bbox = _pick_plate_for_bbox(v_bbox)

            if v_label_norm == "Twowheeler":
                # helmet checks
                helmet_ok = False
                for p in linked_persons:
                    pbox = p.get("bbox")
                    for h in helmets:
                        hb = h.get("bbox")
                        if pbox and hb and _iou(pbox, hb) >= IOU_PERSON_HELMET and (h.get("confidence", 0) >= HELMET_CONF_THRESHOLD):
                            helmet_ok = True
                            break
                    if helmet_ok:
                        break
                if not helmet_ok:
                    violation_types.append("No Helmet")

            elif v_label_norm == "MoreThanTwoWheeler":
                # seatbelt checks
                seatbelt_ok = False
                for p in linked_persons:
                    pbox = p.get("bbox")
                    for s in seatbelts:
                        sb = s.get("bbox")
                        if pbox and sb and _iou(pbox, sb) >= IOU_PERSON_SEATBELT and (s.get("confidence", 0) >= SEATBELT_CONF_THRESHOLD):
                            seatbelt_ok = True
                            break
                    if seatbelt_ok:
                        break
                if not seatbelt_ok:
                    violation_types.append("No Seatbelt")
            else:
                # unknown: fallback to both checks if models exist
                helmet_ok = any(( (_iou(p.get("bbox") or [], h.get("bbox") or []) >= IOU_PERSON_HELMET) and (h.get("confidence",0) >= HELMET_CONF_THRESHOLD) ) for p in linked_persons for h in helmets)
                seatbelt_ok = any(( (_iou(p.get("bbox") or [], s.get("bbox") or []) >= IOU_PERSON_SEATBELT) and (s.get("confidence",0) >= SEATBELT_CONF_THRESHOLD) ) for p in linked_persons for s in seatbelts)
                if helmets and not helmet_ok:
                    violation_types.append("No Helmet")
                if seatbelts and not seatbelt_ok:
                    violation_types.append("No Seatbelt")

            # If plate bbox exists, crop and OCR
            ocr_text = None
            ocr_conf = 0.0
            if plate_bbox:
                plate_bbox_exp = expand_bbox(plate_bbox, frame.shape, pct=0.30)
                crop_path = _save_crop_image(frame, plate_bbox_exp, prefix="violation")
                crop_img = None
                if crop_path:
                    crop_img = cv2.imread(crop_path)
                if crop_img is not None:
                    ocr_text, ocr_conf = extract_plate_text_from_crop(crop_img)

            # if violation -> create annotated screenshot and save
            is_violation = len(violation_types) > 0
            if is_violation:
                annotated = frame.copy()
                # draw vehicle bbox and persons/plate
                draw_bbox_and_label(annotated, v_bbox, f"{v_label_norm} ({v_conf:.2f})", is_violation=True)
                for p in linked_persons:
                    draw_bbox_and_label(annotated, p.get("bbox"), f"person ({p.get('confidence',0):.2f})", is_violation=True)
                if plate_bbox:
                    draw_bbox_and_label(annotated, plate_bbox, f"plate ({plate_model_conf:.2f})", is_violation=True)
                saved_annotated = _save_annotated_image(annotated, prefix="violation_vehicle_annotated")
                saved_crop = _save_crop_image(frame, expand_bbox(plate_bbox or v_bbox, frame.shape, pct=0.25), prefix="violation_vehicle")

            summary = {
                "vehicle_type": v_label_norm,
                "vehicle_label_raw": v_label_raw,
                "vehicle_confidence": float(v_conf),
                "vehicle_bbox": v_bbox,
                "violation": ", ".join(violation_types) if violation_types else None,
                "violation_types": violation_types,
                "number_plate_model_label": plate_model_label,
                "number_plate_model_confidence": float(plate_model_conf),
                "number_plate_bbox": plate_bbox,
                "number_plate_text": ocr_text,
                "number_plate_text_confidence": float(ocr_conf),
                "saved_crop_path": saved_crop,
                "saved_annotated_path": saved_annotated,
                "metadata": {
                    "helmet_conf_global": float(helmet_conf_global),
                    "seatbelt_conf_global": float(seatbelt_conf_global),
                    "anpr_conf_global": float(anpr_conf_global),
                },
                "image_path": str(source_path) if source_path else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            if debug:
                summary["raw_model_outputs"] = model_outputs
            summaries.append(summary)

    else:
        # No vehicles detected: person-level fallback (prefer helmet signal -> Twowheeler)
        for p in persons:
            pbox = p.get("bbox")
            # check helmet overlap
            helmet_found = any(( (_iou(pbox, h.get("bbox") or []) >= IOU_PERSON_HELMET) and (h.get("confidence", 0) >= HELMET_CONF_THRESHOLD) ) for h in helmets)
            seatbelt_found = any(( (_iou(pbox, s.get("bbox") or []) >= IOU_PERSON_SEATBELT) and (s.get("confidence", 0) >= SEATBELT_CONF_THRESHOLD) ) for s in seatbelts)

            inferred_vehicle_type = None
            violation_types = []

            # Prefer inference from helmet model if present
            if helmets:
                inferred_vehicle_type = "Twowheeler"
                if not helmet_found:
                    violation_types.append("No Helmet")
            elif seatbelts:
                inferred_vehicle_type = "MoreThanTwoWheeler"
                if not seatbelt_found:
                    violation_types.append("No Seatbelt")
            else:
                # no supporting models: fallback mark both missing
                inferred_vehicle_type = None
                violation_types.append("No Helmet")
                violation_types.append("No Seatbelt")

            # plate pick
            plate_model_label, plate_model_conf, plate_bbox = (None, 0.0, None)
            if anpr:
                plate_model_label, plate_model_conf, plate_bbox = anpr[0].get("label"), anpr[0].get("confidence", 0.0), anpr[0].get("bbox")

            ocr_text = None; ocr_conf = 0.0; saved_crop = None; saved_annotated = None
            if plate_bbox:
                plate_bbox_exp = expand_bbox(plate_bbox, frame.shape, pct=0.30)
                crop_path = _save_crop_image(frame, plate_bbox_exp, prefix="violation_person")
                crop_img = cv2.imread(crop_path) if crop_path else None
                if crop_img is not None:
                    ocr_text, ocr_conf = extract_plate_text_from_crop(crop_img)

            is_violation = len(violation_types) > 0
            if is_violation:
                annotated = frame.copy()
                draw_bbox_and_label(annotated, pbox, f"person ({p.get('confidence',0):.2f})", is_violation=True)
                if plate_bbox:
                    draw_bbox_and_label(annotated, plate_bbox, f"plate ({plate_model_conf:.2f})", is_violation=True)
                saved_annotated = _save_annotated_image(annotated, prefix="violation_person_annotated")
                saved_crop = _save_crop_image(frame, expand_bbox(plate_bbox or pbox, frame.shape, pct=0.25), prefix="violation_person")

            summary = {
                "vehicle_type": inferred_vehicle_type,
                "vehicle_label_raw": None,
                "vehicle_confidence": None,
                "vehicle_bbox": pbox,
                "violation": ", ".join(violation_types) if violation_types else None,
                "violation_types": violation_types,
                "number_plate_model_label": plate_model_label,
                "number_plate_model_confidence": float(plate_model_conf),
                "number_plate_bbox": plate_bbox,
                "number_plate_text": ocr_text,
                "number_plate_text_confidence": float(ocr_conf),
                "saved_crop_path": saved_crop,
                "saved_annotated_path": saved_annotated,
                "metadata": {
                    "helmet_conf_global": float(helmet_conf_global),
                    "seatbelt_conf_global": float(seatbelt_conf_global),
                    "anpr_conf_global": float(anpr_conf_global),
                },
                "image_path": str(source_path) if source_path else None,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            if debug:
                summary["raw_model_outputs"] = model_outputs
            summaries.append(summary)

    # aggregate annotated image of all summaries (optional)
    annotated_all = frame.copy()
    for s in summaries:
        vb = s.get("vehicle_bbox") or s.get("number_plate_bbox") or s.get("vehicle_bbox")
        lab = s.get("vehicle_type") or s.get("violation") or "entity"
        draw_bbox_and_label(annotated_all, vb, f"{lab}", is_violation=bool(s.get("violation")))
    aggregated_path = _save_annotated_image(annotated_all, prefix="annotated")
    elapsed = time.time() - start_ts
    logger.info("Processing time: %.2f sec -- saved aggregated annotated image: %s", elapsed, aggregated_path)

    return summaries

# -----------------------
# CLI / Test runner
# -----------------------
def run_test_on_image(image_path: Path, device_choice="auto", save_annotated=True, debug=False, show_on_screen=False):
    global models
    # reload models with device choice to ensure device placement
    for name in models:
        if models[name] is not None:
            models[name] = safe_load_model(MODEL_DIR / get_model_filename(name), device_choice=device_choice)

    if not Path(image_path).exists():
        logger.error("Image not found: %s", image_path)
        return None, []

    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("Could not read image: %s", image_path)
        return None, []

    summaries = detect_violations(frame, source_path=str(image_path), device_pref=device_choice, debug=debug)

    # Optionally show annotated aggregated (best effort)
    if show_on_screen:
        try:
            agg = cv2.imread(_save_annotated_image(frame, prefix="temp_display"))
            if agg is not None:
                cv2.imshow("Aggregated", agg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        except Exception:
            logger.debug("Show on screen not available in this environment")

    return summaries

def get_model_filename(model_name):
    filename_map = {
        "vehicle": "vehicle_classifier.pt",
        "person_head": "Person_and_head_detector.pt",
        "helmet": "helmet_detector.pt",
        "seatbelt": "seatbelt_detector.pt",
        "anpr": "anpr_detector.pt",
    }
    return filename_map.get(model_name, "")

def main():
    parser = argparse.ArgumentParser(description="HSDS Inference (helmet/seatbelt/anpr + ocr)")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--video", type=str, default=None, help="Path to video (not fully optimized here)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    device_choice = args.device
    logger.info("HSDS Inference starting. Device=%s EasyOCR=%s", device_choice, _EASYOCR_AVAILABLE)

    if args.image:
        summaries = run_test_on_image(Path(args.image), device_choice=device_choice, debug=args.debug)
        print(json.dumps(summaries, indent=2, default=str))
    elif args.video:
        logger.error("Video mode is not the focus of this script. Use frame sampling + process_video_file if needed.")
    else:
        summaries = run_test_on_image(DEFAULT_TEST_IMAGE, device_choice=device_choice, debug=args.debug)
        print(json.dumps(summaries, indent=2, default=str))

if __name__ == "__main__":
    main()
