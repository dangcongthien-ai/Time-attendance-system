import numpy as np
import argparse
import cv2
import os
import glob

BASE_DIR = "anti_static_images"
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_SKIP = 1

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if face_cascade.empty():
    raise RuntimeError("Khong load duoc Haar Cascade")

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i", "--input",
    type=str,
    default=VIDEOS_DIR,
    help="folder videos (chua real/ fake)"
)
ap.add_argument(
    "-o", "--output",
    type=str,
    default=DATA_DIR,
    help="folder data output (real/ fake)"
)
ap.add_argument(
    "-s", "--skip",
    type=int,
    default=DEFAULT_SKIP
)
args = vars(ap.parse_args())

SCALE_X = 0.85
SCALE_Y = 1.15

total_saved = 0

for label in ["real", "fake"]:
    input_dir = os.path.join(args["input"], label)
    output_dir = os.path.join(args["output"], label)

    if not os.path.isdir(input_dir):
        print(f"[WARN] Khong tim thay folder: {input_dir}")
        continue

    os.makedirs(output_dir, exist_ok=True)

    video_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    if not video_files:
        print(f"[WARN] Khong co video mp4 trong {input_dir}")
        continue

    print(f"\nPROCESS {label.upper()} ({len(video_files)} videos)")

    saved = 0

    for video_path in video_files:
        print(f"Video: {video_path}")
        vs = cv2.VideoCapture(video_path)
        if not vs.isOpened():
            print("Khong mo duoc video")
            continue

        read = 0
        while True:
            ok, frame = vs.read()
            if not ok:
                break

            read += 1
            if read % args["skip"] != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80)
            )

            if len(faces) == 0:
                continue

            # lấy mặt lớn nhất
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

            # CẮT NGANG
            new_w = int(w * SCALE_X)
            dx = (w - new_w) // 2
            startX = max(0, x + dx)
            endX = min(frame.shape[1], startX + new_w)

            # NỚI DỌC
            new_h = int(h * SCALE_Y)
            dy = (new_h - h) // 2
            startY = max(0, y - dy)
            endY = min(frame.shape[0], startY + new_h)

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue

            out_path = os.path.join(
                output_dir, f"frame_{saved:06d}.png"
            )
            cv2.imwrite(out_path, face)
            saved += 1
            total_saved += 1

        vs.release()

    print(f"Saved {saved} images for {label}")

cv2.destroyAllWindows()
print(f"\nHOAN TAT: da luu tong cong {total_saved} khuon mat")
