import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.join("anti_static_images")
DEFAULT_LIVENESS_MODEL_PATH = os.path.join(BASE_DIR, "liveness_torch.pth")
DEFAULT_LIVENESS_LE_PATH = os.path.join(BASE_DIR, "liveness_le.pickle")

class LivenessNet(nn.Module):
    def __init__(self, num_classes=2):
        super(LivenessNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.25),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 3),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LivenessModel:
    def __init__(self,
                 model_path: str = DEFAULT_LIVENESS_MODEL_PATH,
                 le_path: str = DEFAULT_LIVENESS_LE_PATH,
                 device: torch.device | None = None):
        self.model_path = model_path
        self.le_path = le_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: LivenessNet | None = None
        self.le = None
        self.classes: list[str] = []
        self.fake_idx: int | None = None
        self.live_idx: int | None = None

        self._load()

    def _load(self):
        if not (os.path.exists(self.model_path) and os.path.exists(self.le_path)):
            return

        with open(self.le_path, "rb") as f:
            self.le = pickle.load(f)
        self.classes = list(self.le.classes_)

        self.model = LivenessNet(num_classes=len(self.classes))
        state = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.fake_idx = None
        self.live_idx = None
        for i, name in enumerate(self.classes):
            n = name.lower()
            if "fake" in n or "spoof" in n:
                self.fake_idx = i
            if "real" in n or "live" in n:
                self.live_idx = i

        if len(self.classes) == 2:
            if self.fake_idx is None:
                self.fake_idx = 0
            if self.live_idx is None:
                self.live_idx = 1

    def has_model(self) -> bool:
        return self.model is not None and self.le is not None

    def predict(self, face_bgr: np.ndarray):
        if not self.has_model():
            return None
        if face_bgr is None or face_bgr.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (32, 32))
        face = face_rgb.astype("float32") / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        tensor = torch.from_numpy(face).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        label = str(self.classes[idx])
        score = float(probs[idx])

        is_live = True
        if self.fake_idx is not None and idx == self.fake_idx:
            is_live = False
        elif self.live_idx is not None:
            is_live = (idx == self.live_idx)

        return {"label": label, "score": score, "is_live": is_live}
