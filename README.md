# Face Attendance System

Hệ thống chấm công bằng khuôn mặt sử dụng FastAPI, OpenCV Haar Cascade, LBPH Face Recognizer và mô hình liveness detection bằng PyTorch. Dự án hỗ trợ nhận diện nhân viên qua camera, chặn khuôn mặt giả/ảnh tĩnh, ghi nhận check-in/check-out, quản lý người dùng và xem báo cáo chấm công trên giao diện admin.

## Mục lục

- [Tính năng](#tính-năng)
- [Kiến trúc tổng quan](#kiến-trúc-tổng-quan)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Chạy ứng dụng](#chạy-ứng-dụng)
- [Quy trình sử dụng](#quy-trình-sử-dụng)
- [Train model nhận diện nhân viên](#train-model-nhận-diện-nhân-viên)
- [Train model chống giả mạo](#train-model-chống-giả-mạo)
- [Dữ liệu và file runtime](#dữ-liệu-và-file-runtime)
- [API chính](#api-chính)
- [Xử lý lỗi thường gặp](#xử-lý-lỗi-thường-gặp)

## Tính năng

- Nhận diện khuôn mặt nhân viên theo thời gian thực.
- Kiểm tra liveness để giảm trường hợp chấm công bằng ảnh/video giả mạo.
- Tự động quyết định lần quét tiếp theo là `IN` hay `OUT`.
- Chống spam chấm công bằng cooldown.
- Quản lý nhân viên/admin và thu thập ảnh khuôn mặt trên web.
- Dashboard admin gồm số nhân viên, số người có mặt, vắng, đi trễ, về sớm.
- Lịch sử chấm công theo ngày, tính cả người vắng mặt.
- Cấu hình ca làm, thời gian chờ giữa các lần quét, camera source và mật khẩu admin.
- Lưu dữ liệu bằng SQLite, có thể export/đối chiếu bằng file Excel runtime.

## Kiến trúc tổng quan

```text
Camera
  -> Haar Cascade face detection
  -> LivenessModel kiểm tra real/fake
  -> FaceModel nhận diện nhân viên bằng LBPH
  -> AttendanceManager quyết định IN/OUT
  -> SQLite lưu attendance events
  -> Admin UI hiển thị dashboard, users, history, settings
```

Vai trò các module chính:

- `app.py`: khởi tạo FastAPI, camera loop, routes và kết nối các service.
- `attendance_manager.py`: xử lý nghiệp vụ chấm công theo OOP/dataclass.
- `face_model.py`: train/load model LBPH nhận diện nhân viên.
- `anti_static_images/liveness_model.py`: load và predict model thật/giả.
- `anti_static_images/face_extract.py`: tách khuôn mặt từ video real/fake để tạo dataset liveness.
- `anti_static_images/train_liveness.py`: train model liveness PyTorch.
- `database.py`: cấu hình SQLAlchemy engine/session.
- `models.py`: định nghĩa bảng `users`, `attendance`, `settings`, `audit_logs`.
- `services/security.py`: hash mật khẩu admin và ký cookie session.

## Cấu trúc thư mục

```text
.
├── app.py
├── attendance_manager.py
├── config.py
├── database.py
├── face_model.py
├── models.py
├── requirements.txt
├── services/
│   ├── __init__.py
│   └── security.py
├── templates/
│   ├── admin.html
│   ├── index.html
│   └── login.html
├── static/
├── dataset/
└── anti_static_images/
    ├── face_extract.py
    ├── liveness_model.py
    ├── train_liveness.py
    ├── data/
    │   ├── real/
    │   └── fake/
    └── videos/
        ├── real/
        └── fake/
```

## Yêu cầu môi trường

- Python 3.10 trở lên.
- Camera laptop, webcam USB hoặc camera IP/DroidCam.
- Windows/Linux/macOS đều có thể chạy, nhưng dự án hiện đang được phát triển trên Windows.
- Nên dùng môi trường ảo riêng.

Lưu ý: dự án cần `opencv-contrib-python` vì model nhận diện dùng `cv2.face.LBPHFaceRecognizer_create`.

## Cài đặt

Tạo virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài dependencies:

```powershell
pip install -r requirements.txt
```

Nếu đang có `opencv-python` và gặp lỗi `cv2.face`, gợi ý gỡ và cài lại bản contrib:

```powershell
pip uninstall opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

## Cấu hình

Các giá trị mặc định nằm trong `config.py`:

```python
DATASET_DIR = "dataset"
SHIFT_START_TIME = "09:00"
SHIFT_END_TIME = "17:00"
ADMIN_PASSWORD = "admin"
MAX_IMAGES = 40
ADMIN_SESSION_SECONDS = 15 * 60
```

Nên đặt secret riêng cho cookie admin:

```powershell
$env:APP_SECRET_KEY="thay-bang-chuoi-bi-mat-dai-va-ngau-nhien"
```

Có thể tham khảo `.env.example`. Hiện tại app đọc `APP_SECRET_KEY` từ biến môi trường, không tự động load file `.env`.

## Chạy ứng dụng

Chạy trực tiếp:

```powershell
python app.py
```

Hoặc dùng Uvicorn:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Mở trình duyệt:

```text
http://localhost:8000
```

Trang admin:

```text
http://localhost:8000/admin
```

Tài khoản mặc định:

```text
Password: admin
```

Sau khi đăng nhập, nên vào `Settings` đổi mật khẩu admin. Mật khẩu mới sẽ được lưu dạng hash.

## Quy trình sử dụng

1. Chạy ứng dụng.
2. Đăng nhập admin tại `/admin`.
3. Vào `Users`, thêm nhân viên mới.
4. Nhập tên, chọn role, bấm `Capture` để thu thập khuôn mặt.
5. Sau khi capture hoàn tất, bấm `Save`.
6. Hệ thống train lại model LBPH từ ảnh trong `dataset/`.
7. Quay lại màn hình camera chính.
8. Nhân viên đưa mặt vào camera để chấm công.
9. App kiểm tra liveness, nhận diện nhân viên, sau đó ghi `IN` hoặc `OUT`.
10. Admin xem báo cáo trong `Dashboard` và `Attendance History`.

## Train model nhận diện nhân viên

Model nhận diện nhân viên dùng ảnh trong:

```text
dataset/<user_id>/
```

Mỗi user cần có nhiều ảnh khuôn mặt đã tiền xử lý. Khi thêm user bằng giao diện admin, app tự động lưu ảnh vào `dataset/<user_id>/` và train lại model.

File sinh ra:

```text
face_model.yml
labels.npy
threshold.npy
```

Nếu cần train lại thủ công trong Python:

```powershell
python -c "from face_model import FaceModel; m=FaceModel(); m.train(); print('done')"
```

## Train model chống giả mạo

### 1. Chuẩn bị video

Đặt video vào:

```text
anti_static_images/videos/real/
anti_static_images/videos/fake/
```

Trong đó:

- `real`: video mặt thật.
- `fake`: video giả mạo, ví dụ ảnh màn hình, ảnh in, video phát lại.

### 2. Tách khuôn mặt từ video

```powershell
python anti_static_images/face_extract.py
```

Mặc định script sẽ đọc video trong `anti_static_images/videos` và lưu khuôn mặt đã crop vào:

```text
anti_static_images/data/real/
anti_static_images/data/fake/
```

Có thể tùy chỉnh:

```powershell
python anti_static_images/face_extract.py --input anti_static_images/videos --output anti_static_images/data --skip 2
```

### 3. Train liveness model

```powershell
python anti_static_images/train_liveness.py
```

File sinh ra:

```text
anti_static_images/liveness_torch.pth
anti_static_images/liveness_le.pickle
```

Khi app khởi động, `LivenessModel` sẽ tự load hai file này nếu tồn tại.

## Dữ liệu và file runtime

Những file/thư mục sau là dữ liệu runtime hoặc dữ liệu nhạy cảm, không nên commit:

- `sqlite.db`
- `attendance.xlsx`
- `dataset/`
- `face_model.yml`
- `labels.npy`
- `threshold.npy`
- `anti_static_images/data/`
- `anti_static_images/videos/`
- `anti_static_images/*.pth`
- `anti_static_images/*.pickle`

`.gitignore` đã được cấu hình để bỏ qua các file trên.

## API chính

Trang web:

- `GET /`: màn hình camera chấm công.
- `GET /video_feed`: MJPEG camera stream.
- `GET /login`: trang đăng nhập admin.
- `GET /admin`: trang quản trị.

Admin API:

- `GET /api/dashboard`: thống kê tổng quan trong ngày.
- `GET /api/users`: danh sách user.
- `POST /api/users/save`: tạo user mới sau khi capture.
- `PUT /api/users/{user_id}`: cập nhật user.
- `DELETE /api/users/{user_id}`: xóa user.
- `POST /api/capture/start`: bắt đầu capture khuôn mặt.
- `GET /api/capture/status`: tiến độ capture.
- `POST /api/capture/cancel`: hủy capture.
- `GET /api/attendance/history`: báo cáo chấm công theo ngày.
- `GET /api/attendance/details`: chi tiết IN/OUT một ngày.
- `GET /api/settings`: đọc cấu hình hệ thống.
- `POST /api/settings`: cập nhật cấu hình hệ thống.

Các API quản trị yêu cầu cookie admin hợp lệ.

## Xử lý lỗi thường gặp

### `ModuleNotFoundError: No module named 'cv2'`

Chưa cài OpenCV. Chạy:

```powershell
pip install -r requirements.txt
```

### `AttributeError: module 'cv2' has no attribute 'face'`

Đang dùng sai bản OpenCV. Cần `opencv-contrib-python`:

```powershell
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Camera không mở được

Kiểm tra camera source trong `Settings`.

- Webcam mặc định thường là `0`.
- DroidCam/IP camera có thể là URL, ví dụ `http://192.168.1.10:4747/video`.

### Liveness model missing

Train lại model chống giả mạo:

```powershell
python anti_static_images/face_extract.py
python anti_static_images/train_liveness.py
```

### Face model missing

Thêm user và capture khuôn mặt trên admin UI, hoặc train lại:

```powershell
python -c "from face_model import FaceModel; FaceModel().train()"
```

### Đăng nhập admin thất bại sau khi đổi mật khẩu

Mật khẩu admin được lưu trong bảng `settings` của `sqlite.db`. Nếu đây là môi trường demo và cần reset nhanh, có thể xóa `sqlite.db` để app tạo lại DB với mật khẩu mặc định `admin`.

## Ghi chú bảo mật

Đây là đồ án/prototype có thể chạy nội bộ. Nếu triển khai thực tế, nên bổ sung:

- HTTPS.
- Phân quyền chi tiết hơn cho admin.
- Backup database.
- Mã hóa hoặc bảo vệ thư mục chứa dữ liệu khuôn mặt.
- Chính sách lưu trữ/xóa dữ liệu sinh trắc học.
- Log/audit đầy đủ cho các sự kiện nhạy cảm.
