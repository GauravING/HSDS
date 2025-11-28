# inference.py
import sys
import cv2
import json
import torch
import logging
import argparse
import uuid
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime, timezone
import numpy as np

# Optional OCR backends (best-effort import)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    _easyocr_reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA and want GPU OCR
except Exception:
    EASYOCR_AVAILABLE = False
    _easyocr_reader = None

try:
    import pytesseract
    from pytesseract import Output as TESSERACT_OUTPUT
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

# ----------------------------------------------------------
# Logging
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("YOLO-Inference")

# ----------------------------------------------------------
# Paths & config
# ----------------------------------------------------------
MODEL_DIR = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\models")
TEST_IMAGE = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\static\uploads\63f6c10cb87a457b9b59dcbd246a7962_test_img.jpg")
OUTPUT_DIR = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\static\uploads\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE_PREF = "auto"  # "cpu", "cuda", or "auto"

# Colors/styles
BOX_COLOR_VIOLATION = (0, 0, 255)
BOX_COLOR_OK = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# Thresholds (tune as required)
MODEL_CONF_THRESHOLD = 0.10
HELMET_CONF_THRESHOLD = 0.25
SEATBELT_CONF_THRESHOLD = 0.25
HELMET_INFER_FALLBACK = 0.35
ANPR_INFER_THRESHOLD = 0.25

# ----------------------------------------------------------
# DFLoss patch for some ultralytics versions
# ----------------------------------------------------------
try:
    import ultralytics.utils.loss as _uloss
    if not hasattr(_uloss, "DFLoss"):
        import torch.nn as _nn

        class DFLoss(_nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, *a, **k):
                return torch.tensor(0.0)

        setattr(_uloss, "DFLoss", DFLoss)
        logger.info("✅ Patched missing DFLoss in ultralytics.utils.loss")
except Exception as e:
    logger.debug("DFLoss patch skipped or failed: %s", e)

# ----------------------------------------------------------
# Device helper & model loader
# ----------------------------------------------------------
def _choose_device():
    if DEVICE_PREF == "cpu":
        logger.info("🖥️ Forcing CPU")
        return "cpu"
    if DEVICE_PREF == "cuda":
        if torch.cuda.is_available():
            logger.info("⚡ Using CUDA")
            return "cuda"
        else:
            logger.warning("⚠️ CUDA requested but not available; using CPU")
            return "cpu"
    # auto
    if torch.cuda.is_available():
        logger.info("⚡ Auto-detected CUDA")
        return "cuda"
    logger.info("🖥️ Auto-detected CPU")
    return "cpu"

def safe_load_model(path: Path):
    if not path.exists():
        logger.error("❌ Model file not found: %s", path)
        return None
    try:
        model = YOLO(str(path))
        device = _choose_device()
        model.to(device)
        logger.info("✅ Loaded %s on %s", path.name, device.upper())
        return model
    except Exception as e:
        logger.exception("❌ Failed to load model %s: %s", path, e)
        return None

# ----------------------------------------------------------
# Load models (filenames kept as your project uses)
# ----------------------------------------------------------
models = {
    "vehicle": safe_load_model(MODEL_DIR / "vehicle_classifier.pt"),
    "person_head": safe_load_model(MODEL_DIR / "Person_and_head_detector.pt"),
    "helmet": safe_load_model(MODEL_DIR / "helmet_detector.pt"),
    "seatbelt": safe_load_model(MODEL_DIR / "seatbelt_detector.pt"),
    "anpr": safe_load_model(MODEL_DIR / "anpr_detector.pt"),
}

# ----------------------------------------------------------
# Helpers: bbox normalization and IoU
# ----------------------------------------------------------
def _normalize_bbox_global(val):
    if val is None:
        return None
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

# ----------------------------------------------------------
# Run models on a frame and collect detections
# ----------------------------------------------------------
def _run_models_on_frame(frame):
    outputs = {}
    for name, model in models.items():
        if model is None:
            outputs[name] = []
            continue
        try:
            results = model.predict(source=frame, conf=MODEL_CONF_THRESHOLD, imgsz=640, verbose=False)
            dets = []
            for r in results:
                for box in getattr(r, "boxes", []):
                    cls_id = int(getattr(box, "cls", 0))
                    label = model.model.names[cls_id] if hasattr(model, "model") and hasattr(model.model, "names") else str(cls_id)
                    conf = float(getattr(box, "conf", 0.0))
                    xyxy = None
                    if hasattr(box, "xyxy"):
                        try:
                            xyxy = _normalize_bbox_global(getattr(box, "xyxy", None))
                        except Exception:
                            xyxy = None
                    elif hasattr(box, "xywh"):
                        try:
                            vals = box.xywh.tolist()
                            if len(vals) >= 4:
                                x, y, w, h = vals[:4]
                                xyxy = [float(x - w / 2), float(y - h / 2), float(x + w / 2), float(y + h / 2)]
                            else:
                                xyxy = _normalize_bbox_global(getattr(box, "xywh", None))
                        except Exception:
                            xyxy = _normalize_bbox_global(getattr(box, "xywh", None))
                    else:
                        try:
                            xyxy = _normalize_bbox_global(box.tolist())
                        except Exception:
                            xyxy = None
                    dets.append({"label": label, "confidence": round(conf, 4), "bbox": xyxy})
            outputs[name] = dets
        except Exception as e:
            logger.exception("Error running model %s: %s", name, e)
            outputs[name] = []
    return outputs

# ----------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------
def draw_bbox_and_text(frame, bbox, label, is_violation, number_plate, timestamp_str):
    if bbox is None or len(bbox) < 4:
        return
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    color = BOX_COLOR_VIOLATION if is_violation else BOX_COLOR_OK
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    if is_violation:
        text_label = "⚠️ VIOLATION"
    else:
        text_label = "✓ OK"
    if label:
        text_label += f" | {label}"
    if number_plate:
        text_label += f" | Plate: {number_plate}"
    text_label += f" | {timestamp_str}"
    ts = cv2.getTextSize(text_label, FONT, FONT_SCALE, FONT_THICKNESS)[0]
    tx = x1
    ty = y1 - 10
    ty = max(ts[1] + 6, ty)
    cv2.rectangle(frame, (tx, ty - ts[1] - 5), (min(frame.shape[1] - 1, tx + ts[0] + 5), ty + 5), TEXT_BG, -1)
    cv2.putText(frame, text_label, (tx, ty), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

# ----------------------------------------------------------
# OCR helpers (easyocr and pytesseract)
# ----------------------------------------------------------
def _preprocess_plate_for_ocr(img, width_target=320):
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if w == 0:
        return gray
    scale = width_target / float(w)
    new_h = max(16, int(h * scale))
    resized = cv2.resize(gray, (width_target, new_h), interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(resized, 9, 75, 75)
    th = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if np.mean(th) > 127:
        processed = 255 - th
    else:
        processed = th
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    return processed

def extract_plate_text_from_crop(crop_bgr, prefer="both"):
    if crop_bgr is None:
        return None, 0.0
    # Try multiple preprocessing sizes to increase OCR robustness
    tried = []
    easy_text, easy_conf = None, 0.0
    tess_text, tess_conf = None, 0.0

    # Prepare variants: color resized and processed binary
    try:
        # color resized (RGB) for easyocr which prefers color images
        h, w = crop_bgr.shape[:2]
        target_w = 320 if w < 800 else 640
        scale = target_w / float(max(1, w))
        new_h = max(16, int(h * scale))
        resized_color = cv2.resize(crop_bgr, (target_w, new_h), interpolation=cv2.INTER_CUBIC)
        resized_rgb = cv2.cvtColor(resized_color, cv2.COLOR_BGR2RGB)
    except Exception:
        resized_rgb = None

    processed = _preprocess_plate_for_ocr(crop_bgr, width_target=320)

    # Attempt EasyOCR first (if available)
    if EASYOCR_AVAILABLE and _easyocr_reader is not None:
        try:
            # try on color RGB first
            if resized_rgb is not None:
                res = _easyocr_reader.readtext(resized_rgb, detail=1)
                tried.append(('easy_color', bool(res)))
                if res:
                    best = max(res, key=lambda r: r[2])
                    easy_text = best[1].upper().strip()
                    easy_conf = float(best[2])
            # fallback to processed binary
            if (not easy_text) and processed is not None:
                res2 = _easyocr_reader.readtext(processed, detail=1)
                tried.append(('easy_bw', bool(res2)))
                if res2:
                    best = max(res2, key=lambda r: r[2])
                    easy_text = best[1].upper().strip()
                    easy_conf = float(best[2])
        except Exception as e:
            logger.debug("easyocr error: %s", e)

    # Try pytesseract as a fallback (if installed)
    if TESSERACT_AVAILABLE:
        try:
            # Prefer running on the preprocessed image
            if processed is not None:
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                raw = pytesseract.image_to_data(processed, config=custom_config, output_type=TESSERACT_OUTPUT)
                texts = []
                confs = []
                for t, c in zip(raw.get('text', []), raw.get('conf', [])):
                    if t and str(t).strip():
                        texts.append(str(t).strip())
                        try:
                            confs.append(float(c))
                        except Exception:
                            confs.append(0.0)
                if texts:
                    tess_text = "".join(texts).upper().replace(" ", "")
                    valid_confs = [c for c in confs if c >= 0]
                    tess_conf = float(sum(valid_confs) / len(valid_confs)) / 100.0 if valid_confs else 0.0
                    tried.append(('tesseract_bw', True))
            # also try on color resized image if we have it
            if (not tess_text) and resized_color is not None:
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                raw2 = pytesseract.image_to_data(resized_color, config=custom_config, output_type=TESSERACT_OUTPUT)
                texts2 = []
                confs2 = []
                for t, c in zip(raw2.get('text', []), raw2.get('conf', [])):
                    if t and str(t).strip():
                        texts2.append(str(t).strip())
                        try:
                            confs2.append(float(c))
                        except Exception:
                            confs2.append(0.0)
                if texts2:
                    tess_text = "".join(texts2).upper().replace(" ", "")
                    valid_confs2 = [c for c in confs2 if c >= 0]
                    tess_conf = float(sum(valid_confs2) / len(valid_confs2)) / 100.0 if valid_confs2 else 0.0
                    tried.append(('tesseract_color', True))
        except Exception as e:
            logger.debug("pytesseract error: %s", e)

    # Logging to help debug OCR behavior
    if debug_log := False:
        logger.info("OCR tried methods: %s", tried)

    # Decide which result to return based on preferred backend and confidences
    if prefer == "easyocr":
        return (easy_text, easy_conf) if easy_text else (tess_text, tess_conf)
    if prefer == "tesseract":
        return (tess_text, tess_conf) if tess_text else (easy_text, easy_conf)
    # both: choose higher confidence
    if easy_text and tess_text:
        return (easy_text, easy_conf) if easy_conf >= tess_conf else (tess_text, tess_conf)
    if easy_text:
        return easy_text, easy_conf
    if tess_text:
        return tess_text, tess_conf
    return None, 0.0

# ----------------------------------------------------------
# Utility: pick plate overlapping vehicle bbox
# ----------------------------------------------------------
def _pick_plate_for_bbox(vbbox, anpr_list):
    if not anpr_list:
        return None, 0.0, None  # label, conf, bbox
    best_label = None
    best_conf = 0.0
    best_bbox = None
    for a in anpr_list:
        ab = a.get("bbox") or None
        conf = float(a.get("confidence", 0) or 0.0)
        if vbbox and ab:
            i = _iou(vbbox, ab)
            if i > 0.05 and conf > best_conf:
                best_conf = conf
                best_label = a.get("label")
                best_bbox = ab
    if best_label:
        return best_label, best_conf, best_bbox
    # fallback to highest-confidence ANPR
    best = max(anpr_list, key=lambda x: float(x.get("confidence", 0) or 0.0))
    return best.get("label"), float(best.get("confidence", 0) or 0.0), best.get("bbox")

# ----------------------------------------------------------
# Save crops and annotated full image
# ----------------------------------------------------------
def _save_violation_screenshot(frame, vehicle_bbox, prefix, plate_text=None):
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    crop_name = f"{prefix}_crop_{now}_{uid}.jpg"
    full_name = f"{prefix}_annotated_{now}_{uid}.jpg"
    crop_path = str(OUTPUT_DIR / crop_name)
    full_path = str(OUTPUT_DIR / full_name)
    # save full annotated
    try:
        cv2.imwrite(full_path, frame)
    except Exception as e:
        logger.warning("⚠️ Failed to save annotated full image: %s", e)
        full_path = None
    # save crop
    if vehicle_bbox and len(vehicle_bbox) >= 4:
        x1, y1, x2, y2 = [int(max(0, v)) for v in vehicle_bbox[:4]]
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2].copy()
            try:
                cv2.imwrite(crop_path, crop)
            except Exception as e:
                logger.warning("⚠️ Failed to save crop: %s", e)
                crop_path = None
        else:
            crop_path = None
    else:
        crop_path = None
    return crop_path, full_path

# ----------------------------------------------------------
# Hierarchical detection: vehicle -> person -> helmet/seatbelt -> anpr/ocr -> save
# ----------------------------------------------------------
def detect_violations(
    frame,
    db_session=None,
    roi_config=None,
    source_path=None,
    user_id=None,
    debug=False,
    frame_timestamp=None,
    frame_index=None,
):
    """
    Returns a list of JSON-serializable summaries:
    {
      vehicle_type, violation, number_plate (model), number_plate_confidence,
      number_plate_text (OCR), number_plate_text_confidence,
      saved_crop_path, saved_annotated_path, vehicle_bbox, vehicle_confidence, metadata, timestamp
    }
    """
    try:
        model_outputs = _run_models_on_frame(frame)
        timestamp_str = frame_timestamp or datetime.now(timezone.utc).isoformat()
        vehicles = model_outputs.get("vehicle") or []
        persons = model_outputs.get("person_head") or []
        helmets = model_outputs.get("helmet") or []
        seatbelts = model_outputs.get("seatbelt") or []
        anpr = model_outputs.get("anpr") or []

        summaries = []

        # global confidences for fallback logic
        helmet_conf_global = max((h.get("confidence") or 0) for h in helmets) if helmets else 0.0
        seatbelt_conf_global = max((s.get("confidence") or 0) for s in seatbelts) if seatbelts else 0.0
        anpr_conf_global = max((a.get("confidence") or 0) for a in anpr) if anpr else 0.0

        # If the vehicle classifier misses detections entirely, infer a
        # plausible vehicle_type from other model signals so the summary can
        # still state a vehicle type for downstream logic and UI.
        inferred_vehicle = None
        try:
            if not vehicles:
                if helmet_conf_global >= HELMET_INFER_FALLBACK or (helmets and len(helmets) > 0):
                    inferred_vehicle = "Twowheeler"
                elif seatbelt_conf_global >= SEATBELT_CONF_THRESHOLD or anpr_conf_global >= ANPR_INFER_THRESHOLD:
                    inferred_vehicle = "MoreThanTwoWheeler"
                if inferred_vehicle and debug:
                    logger.info("DEBUG: inferred vehicle_type=%s from helmet/seatbelt/anpr signals", inferred_vehicle)
        except Exception:
            inferred_vehicle = None

        if vehicles:
            for v in vehicles:
                raw_label = v.get("label")
                vehicle_bbox = v.get("bbox")
                vehicle_conf = float(v.get("confidence") or 0.0)
                # Determine vehicle_type (exact classes are "Twowheeler" or "MoreThanTwoWheeler")
                vehicle_type = None
                if isinstance(raw_label, str):
                    rl = raw_label.strip().lower()
                    if rl == "twowheeler":
                        vehicle_type = "Twowheeler"
                    elif rl == "morethantwowheeler" or rl == "morethan2wheeler" or rl == "morethan_two_wheeler":
                        vehicle_type = "MoreThanTwoWheeler"
                    else:
                        # fallback mapping by keywords
                        if ("two" in rl and "wheel" in rl) or any(k in rl for k in ("motor", "bike", "scooter", "moped")):
                            vehicle_type = "Twowheeler"
                        elif any(k in rl for k in ("car", "truck", "bus", "van", "auto", "taxi")):
                            vehicle_type = "MoreThanTwoWheeler"
                        else:
                            vehicle_type = None
                else:
                    vehicle_type = None

                # linked persons overlapping this vehicle
                linked_persons = []
                for p in persons:
                    pb = p.get("bbox")
                    if vehicle_bbox and pb and _iou(vehicle_bbox, pb) > 0.05:
                        linked_persons.append(p)

                # fallback inference when vehicle_type unknown
                if vehicle_type is None:
                    if helmet_conf_global >= HELMET_INFER_FALLBACK:
                        vehicle_type = "Twowheeler"
                    elif anpr_conf_global >= ANPR_INFER_THRESHOLD or seatbelt_conf_global >= SEATBELT_CONF_THRESHOLD:
                        vehicle_type = "MoreThanTwoWheeler"

                violation_list = []
                # metadata to include detection confidences / matched boxes
                metadata = {
                    "helmet_confidence_global": float(helmet_conf_global),
                    "seatbelt_confidence_global": float(seatbelt_conf_global),
                    "anpr_confidence_global": float(anpr_conf_global),
                    "helmet_matches": [],
                    "seatbelt_matches": [],
                    "person_matches": [p.get("bbox") for p in linked_persons]
                }

                # Twowheeler -> helmet checks (IoU with person bbox)
                if vehicle_type == "Twowheeler":
                    helmet_ok = False
                    # Look for helmet overlapping any linked person
                    for p in linked_persons:
                        pbox = p.get("bbox")
                        for h in helmets:
                            hb = h.get("bbox")
                            hlabel = (h.get("label") or "").lower()
                            # only count positive helmet detections (not 'no_helmet')
                            helmet_positive = ("helmet" in hlabel) and ("no" not in hlabel)
                            if pbox and hb and _iou(pbox, hb) > 0.15 and helmet_positive and (h.get("confidence", 0) > HELMET_CONF_THRESHOLD):
                                helmet_ok = True
                                metadata["helmet_matches"].append({"bbox": hb, "confidence": float(h.get("confidence", 0)), "label": hlabel})
                                break
                        if helmet_ok:
                            break
                    if not helmet_ok:
                        violation_list.append("No Helmet")

                # MoreThanTwoWheeler -> seatbelt checks
                elif vehicle_type == "MoreThanTwoWheeler":
                    seatbelt_ok = False
                    for p in linked_persons:
                        pbox = p.get("bbox")
                        for s in seatbelts:
                            sb = s.get("bbox")
                            slabel = (s.get("label") or "").lower()
                            seatbelt_positive = ("seatbelt" in slabel) and ("no" not in slabel)
                            if pbox and sb and _iou(pbox, sb) > 0.10 and seatbelt_positive and (s.get("confidence", 0) > SEATBELT_CONF_THRESHOLD):
                                seatbelt_ok = True
                                metadata["seatbelt_matches"].append({"bbox": sb, "confidence": float(s.get("confidence", 0)), "label": slabel})
                                break
                        if seatbelt_ok:
                            break
                    if not seatbelt_ok:
                        violation_list.append("No Seatbelt")

                else:
                    # unknown vehicle -> try both checks on linked persons
                    # consider label semantics too (e.g. 'no_helmet' means no helmet)
                    helmet_ok = any(((_iou(p.get("bbox") or [], h.get("bbox") or []) > 0.15) and (h.get("confidence", 0) > HELMET_CONF_THRESHOLD) and ("helmet" in (h.get("label") or "").lower()) and ("no" not in (h.get("label") or "").lower())) for p in linked_persons for h in helmets)
                    seatbelt_ok = any(((_iou(p.get("bbox") or [], s.get("bbox") or []) > 0.10) and (s.get("confidence", 0) > SEATBELT_CONF_THRESHOLD) and ("seatbelt" in (s.get("label") or "").lower()) and ("no" not in (s.get("label") or "").lower())) for p in linked_persons for s in seatbelts)
                    if linked_persons and not helmet_ok:
                        violation_list.append("No Helmet")
                    if linked_persons and not seatbelt_ok:
                        violation_list.append("No Seatbelt")

                # pick plate (ANPR) that overlaps vehicle bbox
                plate_label, plate_conf, plate_bbox = _pick_plate_for_bbox(vehicle_bbox, anpr)

                # If violation -> prepare annotated image and OCR plate text if available
                saved_crop = None
                saved_full = None
                plate_text = None
                plate_text_conf = 0.0
                if violation_list:
                    annotated = frame.copy()
                    # draw vehicle, persons, helmet/seatbelt boxes
                    draw_bbox_and_text(annotated, vehicle_bbox, vehicle_type or "Unknown", True, plate_label, datetime.now(timezone.utc).isoformat())
                    for p in linked_persons:
                        draw_bbox_and_text(annotated, p.get("bbox"), "person", True, None, datetime.now(timezone.utc).isoformat())
                    for h in helmets:
                        draw_bbox_and_text(annotated, h.get("bbox"), "helmet", False, None, datetime.now(timezone.utc).isoformat())
                    for s in seatbelts:
                        draw_bbox_and_text(annotated, s.get("bbox"), "seatbelt", False, None, datetime.now(timezone.utc).isoformat())
                    # try OCR on plate bbox if present
                    if plate_bbox:
                        x1, y1, x2, y2 = [int(max(0, v)) for v in plate_bbox[:4]]
                        h_img, w_img = frame.shape[:2]
                        x1 = max(0, min(x1, w_img - 1)); x2 = max(0, min(x2, w_img - 1))
                        y1 = max(0, min(y1, h_img - 1)); y2 = max(0, min(y2, h_img - 1))
                        if x2 > x1 and y2 > y1:
                            plate_crop = frame[y1:y2, x1:x2].copy()
                            plate_text, plate_text_conf = extract_plate_text_from_crop(plate_crop, prefer="both")
                            if plate_text:
                                plate_text = plate_text.replace(" ", "")
                                logger.info("📃 OCR plate text: %s (conf %.3f)", plate_text, plate_text_conf)
                    # Save annotated + crop (vehicle bbox)
                    saved_crop, saved_full = _save_violation_screenshot(annotated, vehicle_bbox, prefix="violation_"+(plate_label or "noplate"))
                # Build summary dict
                summary = {
                    "vehicle_type": vehicle_type,
                    "vehicle_label_raw": raw_label,
                    "vehicle_confidence": float(vehicle_conf),
                    "vehicle_bbox": vehicle_bbox,
                    "violation": ", ".join(violation_list) if violation_list else None,
                    "violation_types": violation_list,
                    "number_plate": plate_label,
                    "number_plate_confidence": float(plate_conf) if plate_conf is not None else 0.0,
                    "number_plate_bbox": plate_bbox,
                    "number_plate_text": plate_text,
                    "number_plate_text_confidence": float(plate_text_conf),
                    "saved_crop_path": saved_crop,
                    "saved_annotated_path": saved_full,
                    "metadata": metadata,
                    "image_path": str(source_path) if source_path else None,
                    "timestamp": timestamp_str,
                    "frame_index": frame_index,
                }
                if debug:
                    summary["raw_info"] = model_outputs
                summaries.append(summary)

        else:
            # no vehicle detected -> fallback to person-level checks
            for p in persons:
                pbox = p.get("bbox")
                # Only count positive labels (ignore 'no_helmet' / 'no_seatbelt' predictions)
                helmet_ok = any(((_iou(pbox, h.get("bbox") or []) > 0.15) and (h.get("confidence", 0) > HELMET_CONF_THRESHOLD) and ("helmet" in (h.get("label") or "").lower()) and ("no" not in (h.get("label") or "").lower())) for h in helmets)
                seatbelt_ok = any(((_iou(pbox, s.get("bbox") or []) > 0.10) and (s.get("confidence", 0) > SEATBELT_CONF_THRESHOLD) and ("seatbelt" in (s.get("label") or "").lower()) and ("no" not in (s.get("label") or "").lower())) for s in seatbelts)
                violation_list = []
                # Prefer checks based on inferred vehicle when available
                if inferred_vehicle == "Twowheeler":
                    if not helmet_ok:
                        violation_list.append("No Helmet")
                elif inferred_vehicle == "MoreThanTwoWheeler":
                    if not seatbelt_ok:
                        violation_list.append("No Seatbelt")
                else:
                    if not helmet_ok:
                        violation_list.append("No Helmet")
                    if not seatbelt_ok:
                        violation_list.append("No Seatbelt")
                plate_label = (anpr[0].get("label") if anpr else None)
                plate_conf = float(anpr[0].get("confidence", 0) or 0.0) if anpr else 0.0
                plate_text = None
                plate_text_conf = 0.0
                saved_crop = None
                saved_full = None
                if violation_list:
                    annotated = frame.copy()
                    draw_bbox_and_text(annotated, pbox, inferred_vehicle or "person", True, plate_label, datetime.now(timezone.utc).isoformat())
                    if plate_label:
                        # find bbox of that plate
                        for a in anpr:
                            if a.get("label") == plate_label:
                                ab = a.get("bbox")
                                if ab and len(ab) >= 4:
                                    x1, y1, x2, y2 = [int(max(0, v)) for v in ab[:4]]
                                    h_img, w_img = frame.shape[:2]
                                    x1 = max(0, min(x1, w_img - 1)); x2 = max(0, min(x2, w_img - 1))
                                    y1 = max(0, min(y1, h_img - 1)); y2 = max(0, min(y2, h_img - 1))
                                    if x2 > x1 and y2 > y1:
                                        plate_crop = frame[y1:y2, x1:x2].copy()
                                        plate_text, plate_text_conf = extract_plate_text_from_crop(plate_crop, prefer="both")
                                        if plate_text:
                                            plate_text = plate_text.replace(" ", "")
                                            logger.info("📃 OCR plate text (fallback): %s (conf %.3f)", plate_text, plate_text_conf)
                                        break
                    saved_crop, saved_full = _save_violation_screenshot(annotated, pbox, prefix="violation_person_"+(plate_label or "noplate"))
                summary = {
                    "vehicle_type": inferred_vehicle,
                    "vehicle_label_raw": None,
                    "vehicle_confidence": None,
                    "vehicle_bbox": None,
                    "violation": ", ".join(violation_list) if violation_list else None,
                    "violation_types": violation_list,
                    "number_plate": plate_label,
                    "number_plate_confidence": plate_conf,
                    "number_plate_text": plate_text,
                    "number_plate_text_confidence": float(plate_text_conf),
                    "saved_crop_path": saved_crop,
                    "saved_annotated_path": saved_full,
                    "metadata": {
                        "helmet_confidence_global": float(helmet_conf_global),
                        "seatbelt_confidence_global": float(seatbelt_conf_global),
                        "anpr_confidence_global": float(anpr_conf_global),
                    },
                    "image_path": str(source_path) if source_path else None,
                    "timestamp": timestamp_str,
                    "frame_index": frame_index,
                }
                if debug:
                    summary["raw_info"] = model_outputs
                summaries.append(summary)

        return summaries
    except Exception as e:
        logger.exception("detect_violations failed: %s", e)
        raise

# ----------------------------------------------------------
# Annotate + printing utilities
# ----------------------------------------------------------
def annotate_image_with_violations(frame, summaries, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    for s in summaries:
        bbox = s.get("vehicle_bbox") or None
        vt = s.get("vehicle_type") or "Unknown"
        plate = s.get("number_plate")
        is_violation = bool(s.get("violation"))
        draw_bbox_and_text(frame, bbox, vt, is_violation, plate, timestamp)
    return frame

def print_violation_report(summaries, timestamp_str=None):
    if timestamp_str is None:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    logger.info("\n" + "="*80)
    logger.info("📊 DETECTION REPORT - %s", timestamp_str)
    logger.info("="*80)
    if not summaries:
        logger.info("✅ No detections / no violations")
        logger.info("="*80 + "\n")
        return summaries
    total = len(summaries)
    violations = sum(1 for s in summaries if s.get("violation"))
    logger.info("📈 Total Entities: %d | Violations: %d\n", total, violations)
    for i, s in enumerate(summaries, 1):
        status = "⚠️ VIOLATION" if s.get("violation") else "✅ OK"
        logger.info("\n📍 Entity #%d: %s", i, status)
        logger.info("   Vehicle Type: %s", s.get("vehicle_type"))
        logger.info("   Vehicle Label Raw: %s", s.get("vehicle_label_raw"))
        logger.info("   Vehicle Confidence: %s", s.get("vehicle_confidence"))
        logger.info("   Violation: %s", s.get("violation"))
        logger.info("   Number Plate (model): %s (conf %.3f)", s.get("number_plate"), float(s.get("number_plate_confidence", 0.0)))
        logger.info("   OCR Plate Text: %s (conf %.3f)", s.get("number_plate_text"), float(s.get("number_plate_text_confidence", 0.0)))
        logger.info("   Saved Crop: %s", s.get("saved_crop_path"))
        logger.info("   Saved Annotated: %s", s.get("saved_annotated_path"))
        logger.info("   Timestamp: %s", s.get("timestamp"))
        logger.info("   Metadata: %s", json.dumps(s.get("metadata") or {}, ensure_ascii=False))
    logger.info("\n" + "="*80 + "\n")
    return summaries

# ----------------------------------------------------------
# Test runner functions
# ----------------------------------------------------------
def get_model_filename(model_name):
    return {
        "vehicle": "vehicle_classifier.pt",
        "person_head": "Person_and_head_detector.pt",
        "helmet": "helmet_detector.pt",
        "seatbelt": "seatbelt_detector.pt",
        "anpr": "anpr_detector.pt",
    }.get(model_name, "")

def run_test_on_image(image_path: Path, device_choice="auto", save_annotated=True, debug=False, show_on_screen=True):
    global DEVICE_PREF
    DEVICE_PREF = device_choice
    logger.info("\n🔄 Reloading models with device preference...\n")
    for name in list(models.keys()):
        models[name] = safe_load_model(MODEL_DIR / get_model_filename(name))
    if not image_path.exists():
        logger.error("❌ Image not found: %s", image_path)
        return None, []
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("❌ Could not read image")
        return None, []
    timestamp = datetime.now(timezone.utc)
    summaries = detect_violations(frame, source_path=str(image_path), debug=debug)
    print_violation_report(summaries, timestamp.strftime("%Y-%m-%d %H:%M:%S %Z"))
    annotated = frame.copy()
    annotated = annotate_image_with_violations(annotated, summaries, timestamp.isoformat())
    if save_annotated:
        outp = OUTPUT_DIR / f"annotated_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        try:
            cv2.imwrite(str(outp), annotated)
            logger.info("💾 Annotated image saved: %s", outp)
        except Exception as e:
            logger.warning("⚠️ Could not save annotated image: %s", e)
    if show_on_screen:
        try:
            cv2.imshow("Detection Results", annotated)
            logger.info("📺 Annotated image displayed. Press any key to close window.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning("⚠️ Could not display window (headless?): %s", e)
    return annotated, summaries

def process_video_file(
    path,
    db_session=None,
    frame_skip=5,
    max_frames=None,
    roi_config=None,
    user_id=None,
    debug=False,
    save_annotated=True,
):
    caps = cv2.VideoCapture(str(path))
    if not caps.isOpened():
        raise RuntimeError("Failed to open video: %s" % path)
    results = []
    idx = 0
    processed = 0
    all_frames = []
    try:
        while True:
            ret, frame = caps.read()
            if not ret:
                break
            if idx % max(1, frame_skip) == 0:
                frame_timestamp = datetime.now(timezone.utc)
                frame_timestamp_iso = frame_timestamp.isoformat()
                res = detect_violations(
                    frame,
                    db_session=db_session,
                    roi_config=roi_config,
                    source_path=path,
                    user_id=user_id,
                    debug=debug,
                    frame_timestamp=frame_timestamp_iso,
                    frame_index=idx,
                )
                if res:
                    for summary in res:
                        if summary.get("frame_index") is None:
                            summary["frame_index"] = idx
                        if not summary.get("timestamp"):
                            summary["timestamp"] = frame_timestamp_iso
                    results.extend(res)
                    if save_annotated:
                        all_frames.append((frame.copy(), res))
                processed += 1
                if max_frames and processed >= max_frames:
                    break
            idx += 1
        if save_annotated and all_frames:
            logger.info("💾 Saving annotated frames...")
            for i, (f, summ) in enumerate(all_frames):
                annotated_f = annotate_image_with_violations(f, summ)
                p = OUTPUT_DIR / f"video_frame_{i:04d}.jpg"
                cv2.imwrite(str(p), annotated_f)
            logger.info("✅ Annotated frames saved to %s", OUTPUT_DIR)
        return results
    finally:
        caps.release()

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Hierarchical Helmet & Seatbelt Detection (with OCR)")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--video", type=str, default=None, help="Path to video")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="Device")
    parser.add_argument("--frame-skip", type=int, default=5, help="Video frame skip")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--debug", action="store_true", help="Include raw_info in outputs")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated images")
    args = parser.parse_args()

    global DEVICE_PREF
    DEVICE_PREF = args.device

    logger.info("\n" + "="*80)
    logger.info("🚀 Hierarchical Helmet & Seatbelt Detection (with OCR)")
    logger.info("="*80)
    logger.info("Device: %s", DEVICE_PREF)
    logger.info("Debug: %s", args.debug)
    logger.info("Save annotated: %s", not args.no_save)
    logger.info("="*80 + "\n")

    if args.image:
        run_test_on_image(Path(args.image), device_choice=args.device, save_annotated=not args.no_save, debug=args.debug)
    elif args.video:
        res = process_video_file(Path(args.video), frame_skip=args.frame_skip, max_frames=args.max_frames, debug=args.debug, save_annotated=not args.no_save)
        print_violation_report(res)
    else:
        run_test_on_image(TEST_IMAGE, device_choice=args.device, save_annotated=not args.no_save, debug=args.debug)

if __name__ == "__main__":
    main()
