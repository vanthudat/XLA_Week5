import argparse
from pathlib import Path

import cv2
import numpy as np


# ==================== 1. ĐƯỜNG DẪN ẢNH ĐẦU VÀO ====================
# Ảnh ngôi nhà và thư mục lưu kết quả.
THU_MUC = Path(__file__).resolve().parent
THU_MUC_KET_QUA = THU_MUC / 'ket_qua_cau1'
ANH_DAU_VAO = THU_MUC / 'nha_mau.png'


# ==================== 2. HARRIS TỰ XÂY DỰNG ====================
def harris_tu_viet(anh_xam, block_size=3, k=0.04):
    # Bước 1: Tính đạo hàm theo x và y bằng hai bộ lọc Sobel 3x3.
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T

    # Chuẩn hóa giống Harris OpenCV khi dùng ảnh float trong khoảng [0, 1].
    he_so = 1.0 / (4 * block_size)
    Ix = cv2.filter2D(anh_xam, cv2.CV_32F, sobel_x * he_so)
    Iy = cv2.filter2D(anh_xam, cv2.CV_32F, sobel_y * he_so)

    # Bước 2: Tính các tích đạo hàm tại từng pixel.
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Bước 3: Tính tổng trong cửa sổ block_size x block_size.
    # M = [[Sxx, Sxy], [Sxy, Syy]]
    cua_so = (block_size, block_size)
    Sxx = cv2.boxFilter(Ix2, -1, cua_so, normalize=False)
    Syy = cv2.boxFilter(Iy2, -1, cua_so, normalize=False)
    Sxy = cv2.boxFilter(Ixy, -1, cua_so, normalize=False)

    # Bước 4: Tính độ đáp ứng Harris: R = det(M) - k * trace(M)^2.
    det_M = Sxx * Syy - Sxy * Sxy
    trace_M = Sxx + Syy
    R = det_M - k * trace_M ** 2
    return R


# ==================== 3. LỌC VÀ CHỌN CÁC ĐIỂM GÓC ====================
def lay_diem_goc(R):
    diem_goc = []
    if R.max() <= 0:
        return diem_goc

    # Lọc ngưỡng 1% R lớn nhất và giữ cực đại cục bộ trong vùng 7x7.
    nguong = 0.01 * R.max()
    cuc_dai = cv2.dilate(R, np.ones((7, 7), dtype=np.uint8))
    mat_na = ((R > nguong) & (R == cuc_dai)).astype(np.uint8)

    # Gộp các pixel cực đại liền nhau, lấy một điểm có R lớn nhất mỗi vùng.
    so_vung, nhan, thong_ke, _ = cv2.connectedComponentsWithStats(mat_na)
    for i in range(1, so_vung):  # Bỏ vùng 0 vì đây là nền.
        x, y, rong, cao, _ = thong_ke[i]
        vung_R = R[y:y + cao, x:x + rong].copy()
        vung_nhan = nhan[y:y + cao, x:x + rong]
        vung_R[vung_nhan != i] = -np.inf
        dy, dx = np.unravel_index(np.argmax(vung_R), vung_R.shape)
        diem_goc.append((int(x + dx), int(y + dy)))

    return diem_goc


# ==================== 4. ĐỌC ẢNH VÀ CHUẨN BỊ DỮ LIỆU ====================
if __name__ == '__main__':
    # Đọc tham số dòng lệnh.
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Duong dan anh dau vao')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    # Đọc ảnh ngôi nhà hoặc ảnh được truyền qua --image.
    duong_dan_anh = Path(args.image) if args.image else ANH_DAU_VAO
    if not duong_dan_anh.is_file():
        raise FileNotFoundError(f'Khong tim thay anh: {duong_dan_anh}')

    # Đọc ảnh màu, hỗ trợ đường dẫn có dấu.
    du_lieu = np.fromfile(str(duong_dan_anh), dtype=np.uint8)
    anh = cv2.imdecode(du_lieu, cv2.IMREAD_COLOR)
    if anh is None:
        raise ValueError(f'Khong doc duoc anh: {duong_dan_anh}')

    # Chuyển ảnh xám và chuẩn hóa về float32 trong khoảng [0, 1].
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)
    anh_xam = anh_xam.astype(np.float32) / 255.0

    # ==================== 5. CHẠY HAI CÁCH HARRIS ====================
    # Harris có sẵn: cửa sổ 3x3, Sobel 3x3, k = 0.04.
    R_opencv = cv2.cornerHarris(anh_xam, blockSize=3, ksize=3, k=0.04)

    # Harris tự viết với cùng tham số.
    R_tu_viet = harris_tu_viet(anh_xam)

    # Lấy tọa độ góc từ hai ma trận đáp ứng.
    goc_opencv = lay_diem_goc(R_opencv)
    goc_tu_viet = lay_diem_goc(R_tu_viet)

    # ==================== 6. ĐÁNH DẤU GÓC VÀ GHÉP ẢNH SO SÁNH ====================
    # Đánh dấu góc bằng vòng tròn đỏ trên bản sao của ảnh gốc.
    anh_opencv = anh.copy()
    anh_tu_viet = anh.copy()
    for x, y in goc_opencv:
        cv2.circle(anh_opencv, (x, y), 6, (0, 0, 255), 2, cv2.LINE_AA)
    for x, y in goc_tu_viet:
        cv2.circle(anh_tu_viet, (x, y), 6, (0, 0, 255), 2, cv2.LINE_AA)

    # Ghép ba ảnh theo chiều ngang, thêm tiêu đề và chú thích.
    danh_sach_anh = [anh, anh_opencv, anh_tu_viet]
    tieu_de = ['1. ANH GOC', '2. HARRIS OPENCV', '3. HARRIS TU VIET']
    chu_thich = ['Anh dau vao', 'Goc: vong tron do', 'Goc: vong tron do']
    mau_sac = [(65, 65, 65), (0, 0, 200), (200, 75, 0)]
    anh_co_tieu_de = []
    for i in range(3):
        # Thêm lề trắng và thanh tiêu đề.
        khung = cv2.copyMakeBorder(danh_sach_anh[i], 90, 16, 16, 16,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.rectangle(khung, (0, 0), (khung.shape[1] - 1, 45), mau_sac[i], -1)
        cv2.putText(khung, tieu_de[i], (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(khung, chu_thich[i], (16, 72), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, mau_sac[i], 1, cv2.LINE_AA)
        # Thêm viền xám ngăn cách các ảnh.
        khung = cv2.copyMakeBorder(khung, 12, 12, 12, 12,
                                  cv2.BORDER_CONSTANT, value=(220, 220, 220))
        anh_co_tieu_de.append(khung)
    anh_so_sanh = np.hstack(anh_co_tieu_de)

    # ==================== 7. LƯU ẢNH KẾT QUẢ ====================
    thu_muc = THU_MUC_KET_QUA
    thu_muc.mkdir(exist_ok=True)
    ten_anh = ['anh_goc.png', 'harris_opencv.png',
               'harris_tu_viet.png', 'so_sanh_harris.png']
    cac_anh = [anh, anh_opencv, anh_tu_viet, anh_so_sanh]
    for ten, hinh in zip(ten_anh, cac_anh):
        # Mã hóa PNG và lưu ảnh.
        thanh_cong, du_lieu = cv2.imencode('.png', hinh)
        if not thanh_cong:
            raise RuntimeError(f'Khong luu duoc anh {ten}')
        du_lieu.tofile(str(thu_muc / ten))

    print(f'Da luu anh ket qua vao: {thu_muc}')

    # ==================== 8. HIỂN THỊ KẾT QUẢ ====================
    if args.show:
        # Thu nhỏ cửa sổ hiển thị, giữ nguyên kích thước ảnh lưu.
        cv2.namedWindow('Cau 1 - Harris', cv2.WINDOW_NORMAL)
        chieu_rong = min(1500, anh_so_sanh.shape[1])
        chieu_cao = round(anh_so_sanh.shape[0] * chieu_rong / anh_so_sanh.shape[1])
        cv2.resizeWindow('Cau 1 - Harris', chieu_rong, chieu_cao)
        cv2.imshow('Cau 1 - Harris', anh_so_sanh)
        cv2.waitKey(0)  # Nhấn một phím bất kỳ để đóng cửa sổ.
        cv2.destroyAllWindows()
