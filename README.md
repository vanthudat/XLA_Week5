# Hướng dẫn chạy bài thực hành tuần 5

Phát hiện góc bằng Harris, trích xuất đặc trưng bằng SIFT/ORB và đối sánh hai ảnh bằng ORB.

## 1. Chuẩn bị

Cài Python 3 và bảo đảm lệnh `python` sử dụng được trong terminal. Sau khi tải hoặc giải nén bài làm, mở terminal tại thư mục `Week5` chứa các file `.py` và ảnh đầu vào.

Cài các thư viện vào môi trường Python dùng để chạy bài:

```powershell
python -m pip install opencv-python numpy
```

Các ảnh đầu vào đã có sẵn, giữ nguyên tên và đặt cùng thư mục với code:

- `nha_mau.png`: dùng cho Câu 1 và Câu 2.
- `anhdung.jpg`, `anhnghieng.jpg`: dùng cho Câu 3.

## 2. Chạy từng câu

**Câu 1 — Harris có sẵn và tự viết:**

```powershell
python harris_cau1.py --show
```

Hiển thị ảnh gốc và hai kết quả Harris với các góc được đánh dấu bằng vòng tròn đỏ. Ảnh được lưu trong `ket_qua_cau1/`; mở `so_sanh_harris.png` để xem ảnh so sánh.

**Câu 2 — SIFT và ORB:**

```powershell
python sift_orb_cau2.py --show
```

Hiển thị các keypoints của SIFT và ORB, đồng thời in số keypoints và kích thước ma trận descriptors ra terminal. Ảnh được lưu trong `ket_qua_cau2/`; ảnh so sánh là `so_sanh_sift_orb.png`.

**Câu 3 — Đối sánh ORB, BFMatcher và Ratio Test:**

```powershell
python orb_matching_cau3.py --show
```

Hiển thị đường nối các cặp đặc trưng được giữ sau Ratio Test. Số keypoints và số cặp sau lọc được in ra terminal. Ảnh được lưu tại `ket_qua_cau3/orb_doi_sanh.png`.

Nhấn một phím khi cửa sổ ảnh đang được chọn để đóng. Bỏ `--show` nếu chỉ muốn lưu kết quả. Thư mục kết quả được tạo khi chạy; chạy lại sẽ ghi đè các ảnh kết quả cùng tên.

## 3. Báo cáo

File báo cáo là `main .tex` (có dấu cách trước `.tex`). Biên dịch bằng **XeLaTeX** sau khi chạy đủ ba câu để có ảnh kết quả. Giữ nguyên các thư mục ảnh khi đưa báo cáo lên Overleaf; cài đặt font cần có Times New Roman.

## 4. Nếu không chạy được Python

Kiểm tra:

```powershell
python --version
```

Nếu báo `Python was not found`, cần cài Python và thêm vào PATH. Nếu dùng môi trường ảo, hãy kích hoạt môi trường đó trước và cài thư viện bằng `python -m pip` trong cùng môi trường.

Nếu báo thiếu `cv2` hoặc `numpy`, chạy lại lệnh cài thư viện ở mục 1 bằng đúng môi trường đang chạy code.
