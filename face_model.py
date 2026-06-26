import os
import cv2
import numpy as np
from utils import preprocess_face
from config import DATASET_DIR, MODEL_PATH, LABELS_PATH, THRESHOLD_PATH

class FaceModel:
    """Quản lý mô hình LBPH: train, load, lưu nhãn và ngưỡng."""
    def __init__(self):
        self.dataset_dir = DATASET_DIR
        self.model_path = MODEL_PATH
        self.labels_path = LABELS_PATH
        self.threshold_path = THRESHOLD_PATH

        self.recognizer = None
        self.label_map = {}
        self.threshold = 85

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

    def train(self):
        """Huấn luyện lại mô hình LBPH từ toàn bộ ảnh trong dataset."""
        faces = []
        labels = []
        label_map = {}

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

        person_folders = [
            p for p in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, p))
        ]

        if not person_folders:
            self.recognizer = None
            self.label_map = {}
            self.threshold = 85
            return

        for i, person in enumerate(person_folders):
            path = os.path.join(self.dataset_dir, person)
            label_map[i] = person
            for img_name in os.listdir(path):
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(preprocess_face(img))
                    labels.append(i)

        if not faces:
            self.recognizer = None
            self.label_map = {}
            self.threshold = 85
            return

        recognizer_local = cv2.face.LBPHFaceRecognizer_create(
            radius=2, neighbors=8, grid_x=8, grid_y=8
        )
        recognizer_local.train(faces, np.array(labels))

        confs = []
        for i, img in enumerate(faces):
            pred, conf = recognizer_local.predict(img)
            if pred == labels[i]:
                confs.append(conf)

        if confs:
            mean_conf = np.mean(confs)
            std_conf = np.std(confs)
            threshold_val = mean_conf + std_conf * 2.5
        else:
            threshold_val = 85

        threshold_val = min(max(threshold_val, 70), 130)

        recognizer_local.save(self.model_path)
        np.save(self.labels_path, label_map)
        np.save(self.threshold_path, threshold_val)

        self.recognizer = recognizer_local
        self.label_map = label_map
        self.threshold = threshold_val

    def load_or_train_on_start(self):
        """Load model đã lưu; nếu không có thì train từ dataset nếu sẵn dữ liệu."""
        if (os.path.exists(self.model_path)
                and os.path.exists(self.labels_path)
                and os.path.exists(self.threshold_path)):
            try:
                recognizer_local = cv2.face.LBPHFaceRecognizer_create()
                recognizer_local.read(self.model_path)
                lm = np.load(self.labels_path, allow_pickle=True).item()
                th = np.load(self.threshold_path, allow_pickle=True).item()

                self.recognizer = recognizer_local
                self.label_map = lm
                self.threshold = th
                return
            except Exception:
                self.recognizer = None

        if os.listdir(self.dataset_dir):
            self.train()
        else:
            self.recognizer = None

    def has_model(self) -> bool:
        """Kiểm tra đã có model để nhận diện chưa."""
        return self.recognizer is not None
