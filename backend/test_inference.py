import sys
import cv2
import json
import torch
import logging
from pathlib import Path
from ultralytics import YOLO

# ----------------------------------------------------------
# 🔧 Logger Setup
# ----------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("YOLO-Test")

# ----------------------------------------------------------
# ⚙️ Configuration
# ----------------------------------------------------------
MODEL_DIR = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\models")
TEST_IMAGE = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\static\uploads\63f6c10cb87a457b9b59dcbd246a7962_test_img.jpg")
OUTPUT_IMAGE = Path(r"P:\Helmet_Seatbelt_Detection_system\backend\app\static\uploads\annotated_result.jpg")

# ----------------------------------------------------------
# 🧩 Patch for DFLoss (fixes the AttributeError)
# ----------------------------------------------------------
try:
    import ultralytics.utils.loss as _uloss
    if not hasattr(_uloss, "DFLoss"):
        import torch.nn as _nn

        class DFLoss(_nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, *a, **k):
                return torch.tensor(0.)

        setattr(_uloss, "DFLoss", DFLoss)
        logger.info("✅ Patched missing DFLoss in ultralytics.utils.loss")
except Exception as e:
    logger.warning(f"⚠️ Failed to patch DFLoss: {e}")

# ----------------------------------------------------------
# 🚀 Load models safely
# ----------------------------------------------------------
def safe_load_model(path: Path):
    if not path.exists():
        logger.error(f"❌ Model file not found: {path}")
        return None
    try:
        model = YOLO(str(path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logger.info(f"✅ Loaded {path.name} on {device.upper()}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load model {path.name}: {e}")
        return None

models = {
    "vehicle": safe_load_model(MODEL_DIR / "vehicle_classifier.pt"),
    "person_head": safe_load_model(MODEL_DIR / "Person_and_head_detector.pt"),
    "helmet": safe_load_model(MODEL_DIR / "helmet_detector.pt"),
    "seatbelt": safe_load_model(MODEL_DIR / "seatbelt_detector.pt"),
    "anpr": safe_load_model(MODEL_DIR / "anpr_detector.pt"),
}

# ----------------------------------------------------------
# 🧠 Run YOLO predictions
# ----------------------------------------------------------
def run_yolo_test(image_path: Path):
    if not image_path.exists():
        logger.error(f"❌ Image not found: {image_path}")
        return

    logger.info(f"\n🧠 Running detection on: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        logger.error("❌ Could not read image.")
        return

    all_results = {}

    for name, model in models.items():
        if model is None:
            continue
        logger.info(f"\n🔍 Running {name} model...")
        try:
            results = model.predict(source=frame, conf=0.1, imgsz=640, verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    label = model.model.names[cls_id]
                    conf = float(box.conf)
                    detections.append({
                        "label": label,
                        "confidence": round(conf, 3)
                    })

            if detections:
                logger.info(f"✅ {len(detections)} detections found by {name} model.")
                all_results[name] = detections

                # Draw detections on image
                for r in results:
                    annotated = r.plot()
                cv2.imwrite(str(OUTPUT_IMAGE), annotated)
            else:
                logger.info(f"⚠️ No detections found by {name} model.")
        except Exception as e:
            logger.error(f"❌ Error running {name} model: {e}")

    logger.info("\n📊 Detection Summary:")
    print(json.dumps(all_results, indent=4))

    if all_results:
        logger.info(f"\n🖼️ Annotated output saved at: {OUTPUT_IMAGE}")
    else:
        logger.warning("\n⚠️ No detections at all. Check your model weights or image quality.")

# ----------------------------------------------------------
# 🏁 Main
# ----------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting independent YOLO inference test...\n")
    run_yolo_test(TEST_IMAGE)
    logger.info("\n✅ Test completed.\n")
