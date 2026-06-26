DATASET_DIR = "dataset"

MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.npy"
THRESHOLD_PATH = "threshold.npy"

ATT_MIN_IN_TO_OUT_SECONDS = 60

ATT_MIN_OUT_TO_IN_SECONDS = 60

ATT_TRIGGER_COOLDOWN = 5

OVERLAY_DURATION = 5

SHIFT_START_TIME = "09:00"

SHIFT_END_TIME = "17:00"

ADMIN_PASSWORD = "admin"

MAX_IMAGES = 40

CAM_DEFAULT_DROID_URL = "http://192.168.1.10:4747/video"

ADMIN_SESSION_SECONDS = 15 * 60

FACE_CAPTURE = {
    "active": False,
    "user_id": None,
    "frames": [],
    "count": 0,
    "progress": 0,
    "done": False,
    "error": False,
    "message": ""
}

STRANGER_OVERLAY_DURATION = 2