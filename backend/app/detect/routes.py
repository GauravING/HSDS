# app/detect/routes.py
import os
import uuid
from pathlib import Path
from datetime import datetime
import cv2

from flask import Blueprint, request, current_app, jsonify
import json
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename

from app.db import db
from app.db import crud
from app.services import inference

detect_bp = Blueprint("detect", __name__, url_prefix="/detect")

# Supported file extensions
VIDEO_EXTS = {"mp4", "mov", "avi", "mkv"}
IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}


def allowed_file_ext(filename: str) -> bool:
    """Check if file extension is allowed."""
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in VIDEO_EXTS.union(IMAGE_EXTS)


def is_video_file(filename: str) -> bool:
    """Detect if uploaded file is a video."""
    return filename.rsplit(".", 1)[1].lower() in VIDEO_EXTS


@detect_bp.route("/upload", methods=["POST"])
@jwt_required()
def upload_and_detect():
    """
    Secure endpoint to upload an image or video and run ROI + hierarchical inference.

    Accepts multipart/form-data:
        file: image or video
        frame_skip (optional): skip N frames between detections
        max_frames (optional): process up to N frames from the video
        roi (optional): ROI configuration as JSON (list of polygons or boxes)

    Returns:
        JSON: { "file": filename, "results": [violations] }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file_ext(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    # Save file safely
    upload_folder = Path(current_app.config.get("UPLOAD_FOLDER", "app/static/uploads"))
    upload_folder.mkdir(parents=True, exist_ok=True)

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    saved_path = upload_folder / unique_name

    try:
        file.save(str(saved_path))
    except Exception as e:
        current_app.logger.exception("File saving failed: %s", e)
        return jsonify({"error": "Failed to save file"}), 500

    current_app.logger.debug("Saved upload to %s", str(saved_path))

    # Identify user from JWT
    user_identity = get_jwt_identity()
    user_id = user_identity.get("id") if user_identity else None

    # Optional inference params
    frame_skip = int(request.form.get("frame_skip", 5))
    max_frames = request.form.get("max_frames")
    max_frames = int(max_frames) if max_frames else None

    roi_config = request.form.get("roi")
    if roi_config:
        try:
            import json
            roi_config = json.loads(roi_config)
        except Exception:
            roi_config = None
            current_app.logger.warning("Invalid ROI config provided; skipping ROI-based inference.")

    # Optional debug flag to include raw per-model outputs in the response
    debug_val = request.form.get("debug") or request.args.get("debug")
    debug = False
    if isinstance(debug_val, str) and debug_val.lower() in ("1", "true", "yes"):
        debug = True

    # Optional auto_save flag: when true the route will persist detected
    # violations to the DB automatically. Default is False so the client can
    # decide when to save results (e.g., after user confirmation).
    auto_save_val = request.form.get("auto_save") or request.args.get("auto_save")
    auto_save = False
    if isinstance(auto_save_val, str) and auto_save_val.lower() in ("1", "true", "yes"):
        auto_save = True

    results = []
    start_time = datetime.utcnow()
    try:
        if is_video_file(filename):
            # Process hierarchical + ROI-based detection on video
            current_app.logger.debug("Calling inference.process_video_file on %s", str(saved_path))

            # Quick sanity: ensure OpenCV can open the saved video. Some uploaded
            # videos use codecs/containers not supported by the local OpenCV build.
            # In that case, attempt to transcode the file to a standard H.264 MP4
            # using system `ffmpeg` (if available) and process the transcoded file.
            try:
                vc = cv2.VideoCapture(str(saved_path))
                can_open = vc.isOpened()
                try:
                    vc.release()
                except Exception:
                    pass
            except Exception:
                can_open = False

            processed_path = str(saved_path)
            if not can_open:
                # try to transcode using ffmpeg (if installed)
                import shutil, subprocess

                ffmpeg_path = shutil.which("ffmpeg")
                if ffmpeg_path:
                    transcoded_name = f"transcoded_{saved_path.stem}_{uuid.uuid4().hex}.mp4"
                    transcoded_path = upload_folder / transcoded_name
                    cmd = [ffmpeg_path, "-y", "-i", str(saved_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(transcoded_path)]
                    current_app.logger.info("Attempting ffmpeg transcode: %s", " ".join(cmd))
                    try:
                        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                        if proc.returncode == 0 and transcoded_path.exists():
                            current_app.logger.info("Transcoding succeeded -> %s", str(transcoded_path))
                            processed_path = str(transcoded_path)
                        else:
                            current_app.logger.warning("Transcoding failed (rc=%s): %s", proc.returncode, proc.stderr.decode(errors='ignore'))
                            # continue and let process_video_file attempt to open and fail with a clear error
                    except subprocess.TimeoutExpired:
                        current_app.logger.exception("ffmpeg transcode timed out")
                    except Exception:
                        current_app.logger.exception("ffmpeg transcode failed")
                else:
                    current_app.logger.warning("ffmpeg not found in PATH; cannot transcode uploaded video %s", str(saved_path))

            # Now call the video processor on the original or transcoded file
            current_app.logger.debug("Processing video at %s", processed_path)
            results = inference.process_video_file(
                processed_path,
                db.session,
                frame_skip=frame_skip,
                max_frames=max_frames,
                roi_config=roi_config,
                user_id=user_id,
                debug=debug,
            )
            current_app.logger.debug("process_video_file returned %d summaries", len(results) if results else 0)

            annotated_url = None
        else:
            # Image-based single-frame detection
            current_app.logger.debug("Attempting to read image from %s", str(saved_path))
            frame = cv2.imread(str(saved_path))
            if frame is None:
                current_app.logger.warning("cv2.imread returned None for %s", str(saved_path))
                return jsonify({"error": "Invalid image file (cv2 failed to read)"}), 400

            current_app.logger.debug("Calling inference.detect_violations on image %s", str(saved_path))
            results = inference.detect_violations(
                frame,
                db.session,
                roi_config=roi_config,
                source_path=str(saved_path),
                user_id=user_id,
                debug=debug,
            )
            current_app.logger.debug("detect_violations returned %d summaries", len(results) if results else 0)

            # Annotate and save an annotated image for frontend preview. Do this
            # before normalization (summaries may contain datetimes).
            try:
                annotated = inference.annotate_image_with_violations(frame.copy(), results)
                results_folder = upload_folder / "results"
                results_folder.mkdir(parents=True, exist_ok=True)
                annotated_name = f"annotated_{uuid.uuid4().hex}_{filename}"
                annotated_path = results_folder / annotated_name
                cv2.imwrite(str(annotated_path), annotated)
                # expose annotated image URL relative to Flask static folder
                # annotated_path is under app/static/uploads/...; build /static/ path
                try:
                    static_root = Path(current_app.root_path) / 'app' / 'static'
                    annotated_url = f"/static/{annotated_path.relative_to(static_root).as_posix()}"
                except Exception:
                    annotated_url = None
                current_app.logger.debug("Annotated image saved to %s", str(annotated_path))
            except Exception as e:
                current_app.logger.exception("Failed to create annotated image: %s", e)
                annotated_url = None

    except Exception as e:
        current_app.logger.exception("Inference processing failed: %s", e)
        # Return a bit more detail in dev to help debugging
        resp = {"error": "Inference failed"}
        try:
            resp["detail"] = str(e)
        except Exception:
            pass
        return jsonify(resp), 500

    # Ensure we always have an iterable result from the inference pipeline.
    # Some inference code paths may return None when no detections occur;
    # normalize that to an empty list so downstream code can iterate safely.
    if results is None:
        current_app.logger.warning("Inference returned None; normalizing to empty list")
        results = []

    # Normalize results: ensure required fields exist and are JSON-serializable
    for v in results:
        # Ensure vehicle_type exists
        v["vehicle_type"] = v.get("vehicle_type") or "Unknown"

        # Normalize number_plate
        np_val = v.get("number_plate")
        if isinstance(np_val, dict):
            # sometimes ANPR output may be nested; try common keys
            v["number_plate"] = np_val.get("label") or np_val.get("text") or None
        else:
            v["number_plate"] = np_val

        # Provide a clear 'violation_detected' field for frontend consumption
        if v.get("is_violation"):
            v["violation_detected"] = v.get("violation_type") or "Violation"
        else:
            v["violation_detected"] = "No Violation"

        # Ensure timestamp is ISO string
        ts = v.get("timestamp")
        try:
            if hasattr(ts, "isoformat"):
                v["timestamp"] = ts.isoformat()
            elif isinstance(ts, (int, float)):
                # epoch -> iso
                v["timestamp"] = datetime.utcfromtimestamp(float(ts)).isoformat()
            else:
                v["timestamp"] = str(ts)
        except Exception:
            v["timestamp"] = str(ts)

        # Ensure confidence_score exists
        v["confidence_score"] = v.get("confidence_score")

    # Optionally auto-log detected violations in DB. When `auto_save` is
    # False we skip persistence and mark saved=False so the frontend can show
    # results and let the user choose when to save.
    if auto_save:
        for v in results:
            try:
                if v.get("violation_id"):
                    # inference already persisted this one
                    v["saved"] = True
                    continue
                # Only persist actual violations (is_violation True).
                if not v.get("is_violation"):
                    v["saved"] = False
                    continue

                db_obj = crud.log_violation(
                    db.session,
                    number_plate=v.get("number_plate"),
                    violation_type=v.get("violation_type"),
                    vehicle_type=v.get("vehicle_type"),
                    confidence_score=v.get("confidence_score"),
                    location=v.get("location"),
                    image_path=v.get("image_path", str(saved_path)),
                    video_clip_path=v.get("video_clip_path"),
                    status="Pending Review",
                    # Use server-side timestamp to avoid client-format issues
                    timestamp=datetime.utcnow(),
                )
                v["saved"] = True
                v["violation_id"] = getattr(db_obj, "violation_id", None)
            except Exception as e:
                # attach error to the response so frontend and tests can see it
                current_app.logger.exception("Failed to log violation: %s", e)
                v["saved"] = False
                try:
                    v["save_error"] = str(e)
                except Exception:
                    v["save_error"] = "unknown error"
    else:
        # mark all returned results as not saved so the UI can offer an
        # explicit save action to the user
        for v in results:
            v["saved"] = False

    # Finalize timing and response
    end_time = datetime.utcnow()
    processing_time = (end_time - start_time).total_seconds()

    resp = {"file": unique_name, "results": results, "processing_time": f"{processing_time:.2f} seconds"}
    try:
        if 'annotated_url' in locals() and annotated_url:
            resp["annotated_image"] = annotated_url
    except Exception:
        pass

    return jsonify(resp), 200


@detect_bp.route("/debug", methods=["GET", "POST"])
def debug_detect():
    """Lightweight debug endpoint: accepts a single image file (no JWT),
    runs detection with debug=True and returns raw per-model outputs. This
    does NOT save anything to the DB or write annotated images.
    """
    # If accessed via GET, return a tiny HTML form so you can test from the browser.
    if request.method == "GET":
        return (
            "<html><body>"
            "<h3>Debug Detect</h3>"
            "<form method=\"post\" enctype=\"multipart/form-data\">"
            "<input type=\"file\" name=\"file\"/>"
            "<input type=\"submit\" value=\"Upload\"/>"
            "</form></body></html>",
            200,
            {"Content-Type": "text/html"},
        )

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save to a temp path so cv2 can read it
    upload_folder = Path(current_app.config.get("UPLOAD_FOLDER", "app/static/uploads"))
    upload_folder.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    tmp_path = upload_folder / f"debug_{uuid.uuid4().hex}_{filename}"
    try:
        file.save(str(tmp_path))
    except Exception as e:
        current_app.logger.exception("Debug file saving failed: %s", e)
        return jsonify({"error": "Failed to save file"}), 500

    frame = cv2.imread(str(tmp_path))
    if frame is None:
        return jsonify({"error": "Invalid image file (cv2 failed to read)"}), 400

    try:
        results = inference.detect_violations(frame, db=None, roi_config=None, source_path=str(tmp_path), user_id=None, debug=True)
        return jsonify({"file": filename, "results": results}), 200
    except Exception as e:
        current_app.logger.exception("Debug detection failed: %s", e)
        return jsonify({"error": "Detection failed", "detail": str(e)}), 500


@detect_bp.route("/raw_debug", methods=["POST"])
def raw_debug():
    """Return raw per-model outputs and final summaries for a single uploaded image.

    This endpoint is unauthenticated and intended for debugging: it returns the
    model-level detections (vehicle/helmet/seatbelt/person_head/anpr) and the
    final summaries produced by `detect_violations` (with debug=True).
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    upload_folder = Path(current_app.config.get("UPLOAD_FOLDER", "app/static/uploads"))
    upload_folder.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    tmp_path = upload_folder / f"raw_debug_{uuid.uuid4().hex}_{filename}"
    try:
        file.save(str(tmp_path))
    except Exception as e:
        current_app.logger.exception("Raw debug file saving failed: %s", e)
        return jsonify({"error": "Failed to save file"}), 500

    frame = cv2.imread(str(tmp_path))
    if frame is None:
        return jsonify({"error": "Invalid image file (cv2 failed to read)"}), 400

    try:
        # Get raw per-model outputs
        model_outputs = inference._run_models_on_frame(frame)

        # Get summaries (with debug=True so raw_info is attached)
        summaries = inference.detect_violations(frame, db=None, roi_config=None, source_path=str(tmp_path), user_id=None, debug=True)

        return jsonify({"file": filename, "model_outputs": model_outputs, "summaries": summaries}), 200
    except Exception as e:
        current_app.logger.exception("Raw debug failed: %s", e)
        return jsonify({"error": "Raw debug failed", "detail": str(e)}), 500


@detect_bp.route("/raw_video_debug", methods=["POST"])
def raw_video_debug():
    """Unauthenticated helper for debugging video uploads.

    Saves the uploaded video to the uploads folder, tries to transcode if
    OpenCV can't open it (using system ffmpeg if available), then runs
    `inference.process_video_file` with conservative defaults and returns
    the raw summaries. This is intended for debugging and should not be
    used in production without auth.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    upload_folder = Path(current_app.config.get("UPLOAD_FOLDER", "app/static/uploads"))
    upload_folder.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    tmp_path = upload_folder / f"raw_video_debug_{uuid.uuid4().hex}_{filename}"
    try:
        file.save(str(tmp_path))
    except Exception as e:
        current_app.logger.exception("Raw video debug file saving failed: %s", e)
        return jsonify({"error": "Failed to save file"}), 500

    # Basic check: can OpenCV open it? If not, try ffmpeg transcode (best-effort)
    try:
        vc = cv2.VideoCapture(str(tmp_path))
        can_open = vc.isOpened()
        try:
            vc.release()
        except Exception:
            pass
    except Exception:
        can_open = False

    processed_path = str(tmp_path)
    if not can_open:
        import shutil, subprocess
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            transcoded_name = f"transcoded_{tmp_path.stem}_{uuid.uuid4().hex}.mp4"
            transcoded_path = upload_folder / transcoded_name
            cmd = [ffmpeg_path, "-y", "-i", str(tmp_path), "-c:v", "libx264", "-preset", "fast", "-crf", "23", str(transcoded_path)]
            current_app.logger.info("Raw video debug: attempting ffmpeg transcode: %s", " ".join(cmd))
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
                if proc.returncode == 0 and transcoded_path.exists():
                    current_app.logger.info("Raw video debug: transcoding succeeded -> %s", str(transcoded_path))
                    processed_path = str(transcoded_path)
                else:
                    current_app.logger.warning("Raw video debug: transcoding failed (rc=%s): %s", proc.returncode, proc.stderr.decode(errors='ignore'))
            except subprocess.TimeoutExpired:
                current_app.logger.exception("Raw video debug: ffmpeg transcode timed out")
            except Exception:
                current_app.logger.exception("Raw video debug: ffmpeg transcode failed")
        else:
            current_app.logger.warning("Raw video debug: ffmpeg not found in PATH; attempting to process original file")

    # Run the video processor with conservative limits so debugging is fast
    try:
        summaries = inference.process_video_file(processed_path, db_session=None, frame_skip=int(request.form.get('frame_skip', 5)), max_frames=int(request.form.get('max_frames', 30)), roi_config=None, user_id=None, debug=True, save_annotated=False)
        # Also return the raw model outputs for the first frame if possible
        first_model_outputs = None
        try:
            # open first frame to run models directly
            vc2 = cv2.VideoCapture(processed_path)
            ok, first_frame = vc2.read()
            try:
                vc2.release()
            except Exception:
                pass
            if ok and first_frame is not None:
                first_model_outputs = inference._run_models_on_frame(first_frame)
        except Exception:
            first_model_outputs = None

        return jsonify({"file": filename, "processed_path": processed_path, "first_model_outputs": first_model_outputs, "summaries": summaries}), 200
    except Exception as e:
        current_app.logger.exception("Raw video debug processing failed: %s", e)
        return jsonify({"error": "Video processing failed", "detail": str(e)}), 500
