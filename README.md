# Face Attendance System

Hệ thống chấm công bằng khuôn mặt sử dụng FastAPI, OpenCV, LBPH và mô hình liveness PyTorch. Hệ thống hỗ trợ nhận diện nhân viên, kiểm tra khuôn mặt thật/giả, ghi nhận `IN`/`OUT` và quản lý dữ liệu chấm công.

## Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Cấu hình

Các giá trị mặc định nằm trong `config.py`, gồm:

- Thời gian bắt đầu và kết thúc ca làm.
- Nguồn camera.
- Số ảnh cần thu thập cho mỗi nhân viên.
- Mật khẩu admin mặc định: `admin`.

Các cấu hình này cũng có thể thay đổi trong trang `Settings` của admin.

## Chạy ứng dụng

```powershell
python app.py
```

Ứng dụng tự mở trang chấm công mặc định:

```text
http://localhost:8000/attendance
```

Trang quản trị:

```text
http://localhost:8000/admin
```

Đăng nhập lần đầu bằng mật khẩu `admin`.

## Quy trình sử dụng

1. Chạy ứng dụng bằng `python app.py`.
2. Mở trang admin và đăng nhập.
3. Vào `Users`, nhập thông tin nhân viên và bắt đầu chụp khuôn mặt.
4. Lưu nhân viên sau khi thu thập đủ ảnh; hệ thống tự train lại model nhận diện.
5. Mở `http://localhost:8000/attendance` để chấm công.
6. Nhân viên nhìn vào camera; hệ thống kiểm tra liveness, nhận diện và ghi `IN` hoặc `OUT`.
7. Xem thống kê và lịch sử chấm công trong trang admin.

## Train model nhận diện

Ảnh khuôn mặt của mỗi nhân viên được lưu tại:

```text
dataset/<user_id>/
```

Khi thêm nhân viên trên trang admin, hệ thống tự train lại model. Có thể train thủ công bằng lệnh:

```powershell
python -c "from face_model import FaceModel; FaceModel().train()"
```

Các file model được tạo:

```text
face_model.yml
labels.npy
threshold.npy
```

## Train model chống giả mạo

1. Đặt video khuôn mặt thật vào `anti_static_images/videos/real/`.
2. Đặt video giả mạo vào `anti_static_images/videos/fake/`.
3. Tách khuôn mặt từ video:

```powershell
python anti_static_images/face_extract.py
```

4. Train model liveness:

```powershell
python anti_static_images/train_liveness.py
```

Các file model được tạo:

```text
anti_static_images/liveness_torch.pth
anti_static_images/liveness_le.pickle
```

Khởi động lại ứng dụng sau khi train để sử dụng model mới.
