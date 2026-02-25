"""
UAV-Based Waste Detection Web System
======================================
Production-ready Flask app for Render.com deployment
Database : Supabase PostgreSQL
Model    : YOLOv8 + SAHI (loaded from Google Drive if missing)
"""

import os
import logging
import traceback
import uuid
from datetime import datetime, timezone

import cv2
import gdown
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ─────────────────────────────────────────────
# Load .env file (local dev only — Render uses its own env vars)
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)

UPLOAD_FOLDER      = os.path.join("static", "uploads")
WEIGHTS_PATH       = os.path.join("weights", "best.pt")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

os.makedirs(UPLOAD_FOLDER,        exist_ok=True)
os.makedirs("weights",            exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Valid Class Names — update to match your model
# ─────────────────────────────────────────────
VALID_CLASSES = [
    "plastic_bottle",
    "styrofoam",
    "plastic_bag",
    "can",
    "cardboard",
    "glass",
    "organic_waste",
    "other_waste",
]

# ─────────────────────────────────────────────
# Model Download from Google Drive (if missing)
# ─────────────────────────────────────────────
# Set GDRIVE_MODEL_ID in your environment/.env file
# Get it from your Google Drive share link:
#   https://drive.google.com/file/d/FILE_ID_HERE/view
GDRIVE_MODEL_ID = os.environ.get("14gzFlrIcwXQTPFndRPYAyKwIEQlEi3UD", "")

def download_model_if_missing():
    """Download best.pt from Google Drive if not present locally."""
    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Model found at {WEIGHTS_PATH}")
        return

    if not GDRIVE_MODEL_ID:
        logger.error("GDRIVE_MODEL_ID not set and model not found. Cannot start.")
        raise FileNotFoundError(
            "Model weights not found. Set GDRIVE_MODEL_ID environment variable."
        )

    logger.info("Model not found locally. Downloading from Google Drive...")
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}",
            WEIGHTS_PATH,
            quiet=False,
        )
        logger.info(f"✅ Model downloaded to {WEIGHTS_PATH}")
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        raise

download_model_if_missing()

# ─────────────────────────────────────────────
# PostgreSQL — Supabase Connection
# ─────────────────────────────────────────────
# Set DATABASE_URL in .env (local) or Render environment variables (production)
#
# Supabase connection string format:
#   postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
#
DATABASE_URL = os.environ.get("postgresql://postgres:Piyashtp@2001@db.jdnewlrwnodoipgvbpmz.supabase.co:5432/postgres", "")

if not DATABASE_URL:
    raise EnvironmentError(
        "DATABASE_URL environment variable is not set. "
        "Add it to your .env file or Render environment variables."
    )


def get_db_connection():
    """Create and return a new PostgreSQL connection per request."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    return conn


def init_db():
    """
    Create annotations table and indexes on startup.
    Safe to run multiple times — uses IF NOT EXISTS.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS annotations (
        id             SERIAL PRIMARY KEY,
        image_name     VARCHAR(255)   NOT NULL,
        image_path     TEXT           NOT NULL,
        class_name     VARCHAR(100)   NOT NULL,
        class_id       INTEGER        NOT NULL,

        bbox_x         INTEGER        NOT NULL,
        bbox_y         INTEGER        NOT NULL,
        bbox_width     INTEGER        NOT NULL,
        bbox_height    INTEGER        NOT NULL,

        yolo_x_center  NUMERIC(10,6)  NOT NULL,
        yolo_y_center  NUMERIC(10,6)  NOT NULL,
        yolo_width     NUMERIC(10,6)  NOT NULL,
        yolo_height    NUMERIC(10,6)  NOT NULL,

        image_width    INTEGER        NOT NULL,
        image_height   INTEGER        NOT NULL,

        annotated_by   VARCHAR(100)   DEFAULT 'anonymous',
        created_at     TIMESTAMPTZ    DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_ann_image_name ON annotations (image_name);
    CREATE INDEX IF NOT EXISTS idx_ann_class_name ON annotations (class_name);
    CREATE INDEX IF NOT EXISTS idx_ann_created_at ON annotations (created_at);
    """
    conn = get_db_connection()
    cur  = conn.cursor()
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("✅ Database table 'annotations' ready.")


try:
    init_db()
    logger.info("✅ Supabase PostgreSQL connected.")
except Exception as e:
    logger.error(f"❌ Database init failed: {e}")
    raise

# ─────────────────────────────────────────────
# SAHI Detection Model — Loaded Once at Startup
# ─────────────────────────────────────────────
def load_detection_model(weights_path: str):
    """Load YOLOv8 model via SAHI. Uses GPU if available, falls back to CPU."""
    try:
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=weights_path,
            confidence_threshold=0.25,
            device="cuda",
        )
        logger.info("✅ Model loaded on CUDA.")
        return model
    except Exception:
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=weights_path,
            confidence_threshold=0.25,
            device="cpu",
        )
        logger.info("✅ Model loaded on CPU.")
        return model


detection_model = load_detection_model(WEIGHTS_PATH)

# ─────────────────────────────────────────────
# YOLO BBox Conversion
# ─────────────────────────────────────────────
def convert_to_yolo_format(
    x_min: float, y_min: float,
    box_w: float, box_h: float,
    img_w: int,   img_h: int,
) -> dict:
    """
    Convert pixel bounding box to normalized YOLO format.
    Input : x_min, y_min, width, height (pixels)
    Output: x_center, y_center, width, height (0.0 – 1.0)
    """
    return {
        "x_center": round((x_min + box_w / 2) / img_w, 6),
        "y_center": round((y_min + box_h / 2) / img_h, 6),
        "width":    round(box_w / img_w,                 6),
        "height":   round(box_h / img_h,                 6),
    }


def validate_yolo_bbox(bbox: dict) -> bool:
    return all(0.0 <= v <= 1.0 for v in bbox.values())

# ─────────────────────────────────────────────
# Inference Helpers
# ─────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def run_sliced_inference(image_path: str):
    """Run SAHI sliced inference on a large drone image."""
    return get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.25,
        overlap_width_ratio=0.25,
        perform_standard_pred=True,
        postprocess_type="NMM",
        postprocess_match_threshold=0.5,
        verbose=0,
    )


def save_annotated_image(image_path: str, result, output_path: str):
    """Draw bounding boxes on image and save annotated output."""
    image = cv2.imread(image_path)
    for pred in result.object_prediction_list:
        bbox  = pred.bbox
        x1, y1, x2, y2 = (
            int(bbox.minx), int(bbox.miny),
            int(bbox.maxx), int(bbox.maxy)
        )
        label = pred.category.name
        score = pred.score.value
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(output_path, image)


def build_detection_response(result, original_url: str, annotated_url: str) -> dict:
    detections = []
    for idx, pred in enumerate(result.object_prediction_list):
        bbox = pred.bbox
        x1, y1, x2, y2 = (
            int(bbox.minx), int(bbox.miny),
            int(bbox.maxx), int(bbox.maxy)
        )
        detections.append({
            "id":         idx + 1,
            "class":      pred.category.name,
            "confidence": round(pred.score.value, 4),
            "bbox": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width": x2 - x1, "height": y2 - y1,
            },
        })
    return {
        "success":             True,
        "total_detections":    len(detections),
        "original_image_url":  original_url,
        "annotated_image_url": annotated_url,
        "detections":          detections,
    }

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", valid_classes=VALID_CLASSES)


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type."}), 400

    try:
        unique_id     = str(uuid.uuid4())
        ext           = file.filename.rsplit(".", 1)[1].lower()
        original_name = f"{unique_id}.{ext}"
        original_path = os.path.join(app.config["UPLOAD_FOLDER"], original_name)
        file.save(original_path)

        result = run_sliced_inference(original_path)

        if len(result.object_prediction_list) == 0:
            return jsonify({
                "success":             True,
                "total_detections":    0,
                "message":             "No waste objects detected.",
                "original_image_url":  f"/static/uploads/{original_name}",
                "annotated_image_url": None,
                "image_filename":      original_name,
                "detections":          [],
            })

        annotated_name = f"detected_{unique_id}.{ext}"
        annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], annotated_name)
        save_annotated_image(original_path, result, annotated_path)

        response = build_detection_response(
            result,
            f"/static/uploads/{original_name}",
            f"/static/uploads/{annotated_name}",
        )
        response["image_filename"] = original_name
        return jsonify(response)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    """
    Accept user annotation and store in Supabase PostgreSQL.
    No files created. No model modified.

    Expected JSON:
    {
        "image_name":   "abc123.jpg",
        "class_name":   "plastic_bottle",
        "x":            120,
        "y":            80,
        "width":        200,
        "height":       150,
        "annotated_by": "user_001"   (optional)
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid or empty JSON body."}), 400

    # Validate required fields
    required = ["image_name", "class_name", "x", "y", "width", "height"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"success": False,
                        "error": f"Missing fields: {missing}"}), 400

    image_name = str(data["image_name"]).strip()
    class_name = str(data["class_name"]).strip().lower()

    if class_name not in VALID_CLASSES:
        return jsonify({"success": False,
                        "error": f"Invalid class. Valid: {VALID_CLASSES}"}), 400

    try:
        x     = float(data["x"])
        y     = float(data["y"])
        box_w = float(data["width"])
        box_h = float(data["height"])
    except (TypeError, ValueError):
        return jsonify({"success": False,
                        "error": "Bounding box values must be numeric."}), 400

    if box_w <= 0 or box_h <= 0:
        return jsonify({"success": False,
                        "error": "Width and height must be > 0."}), 400

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)
    if not os.path.exists(image_path):
        return jsonify({"success": False,
                        "error": f"Image '{image_name}' not found."}), 404

    img = cv2.imread(image_path)
    if img is None:
        return jsonify({"success": False,
                        "error": "Cannot read image file."}), 500

    img_h, img_w = img.shape[:2]

    if x < 0 or y < 0 or (x + box_w) > img_w or (y + box_h) > img_h:
        return jsonify({"success": False,
                        "error": f"Bbox exceeds image bounds ({img_w}×{img_h})."}), 400

    bbox_yolo = convert_to_yolo_format(x, y, box_w, box_h, img_w, img_h)
    if not validate_yolo_bbox(bbox_yolo):
        return jsonify({"success": False,
                        "error": "YOLO bbox out of [0,1] range."}), 400

    insert_sql = """
        INSERT INTO annotations (
            image_name, image_path, class_name, class_id,
            bbox_x, bbox_y, bbox_width, bbox_height,
            yolo_x_center, yolo_y_center, yolo_width, yolo_height,
            image_width, image_height, annotated_by, created_at
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s, %s
        ) RETURNING id;
    """

    try:
        conn = get_db_connection()
        cur  = conn.cursor()
        cur.execute(insert_sql, (
            image_name, image_path.replace("\\", "/"),
            class_name, VALID_CLASSES.index(class_name),
            int(x), int(y), int(box_w), int(box_h),
            bbox_yolo["x_center"], bbox_yolo["y_center"],
            bbox_yolo["width"],    bbox_yolo["height"],
            img_w, img_h,
            str(data.get("annotated_by", "anonymous")).strip(),
            datetime.now(timezone.utc),
        ))
        new_id = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM annotations;")
        total  = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Annotation saved | id={new_id} | {class_name} | {image_name}")

        return jsonify({
            "success":           True,
            "message":           "Annotation saved to database.",
            "annotation_id":     new_id,
            "class_name":        class_name,
            "class_id":          VALID_CLASSES.index(class_name),
            "bbox_yolo":         bbox_yolo,
            "total_annotations": total,
        })

    except psycopg2.Error as e:
        logger.error(f"DB error: {e}")
        return jsonify({"success": False,
                        "error": f"Database error: {str(e)}"}), 500


@app.route("/annotation_stats", methods=["GET"])
def annotation_stats():
    """Return total count, per-class breakdown, and 5 most recent annotations."""
    try:
        conn = get_db_connection()
        cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        cur.execute("SELECT COUNT(*) AS total FROM annotations;")
        total = cur.fetchone()["total"]

        cur.execute("""
            SELECT class_name, COUNT(*) AS count
            FROM annotations
            GROUP BY class_name
            ORDER BY count DESC;
        """)
        by_class = {r["class_name"]: r["count"] for r in cur.fetchall()}

        cur.execute("""
            SELECT id, image_name, class_name, annotated_by, created_at
            FROM annotations
            ORDER BY created_at DESC LIMIT 5;
        """)
        recent = [{
            "id":           r["id"],
            "image_name":   r["image_name"],
            "class_name":   r["class_name"],
            "annotated_by": r["annotated_by"],
            "created_at":   r["created_at"].isoformat(),
        } for r in cur.fetchall()]

        cur.close()
        conn.close()

        return jsonify({
            "success":       True,
            "total":         total,
            "by_class":      by_class,
            "recent":        recent,
            "valid_classes": VALID_CLASSES,
        })

    except psycopg2.Error as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
