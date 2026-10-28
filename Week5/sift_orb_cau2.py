import argparse
from pathlib import Path

import cv2
import numpy as np


# ==================== 1. ĐƯỜNG DẪN ẢNH ĐẦU VÀO ====================
THU_MUC = Path(__file__).resolve().parent
ANH_DAU_VAO = THU_MUC / 'nha_mau.png'
THU_MUC_KET_QUA = THU_MUC / 'ket_qua_cau2'


if __name__ == '__main__':
    # Đọc tham số dòng lệnh.
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Duong dan anh dau vao')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    # ==================== 2. ĐỌC ẢNH VÀ CHUYỂN ẢNH XÁM ====================
    duong_dan_anh = Path(args.image) if args.image else ANH_DAU_VAO
    if not duong_dan_anh.is_file():
        raise FileNotFoundError(f'Khong tim thay anh: {duong_dan_anh}')

    # Đọc ảnh màu, hỗ trợ đường dẫn có dấu.
    du_lieu = np.fromfile(str(duong_dan_anh), dtype=np.uint8)
    anh = cv2.imdecode(du_lieu, cv2.IMREAD_COLOR)
    if anh is None:
        raise ValueError(f'Khong doc duoc anh: {duong_dan_anh}')

    # SIFT và ORB nhận ảnh xám kiểu uint8.
    anh_xam = cv2.cvtColor(anh, cv2.COLOR_BGR2GRAY)

    # ==================== 3. PHÁT HIỆN VÀ MÔ TẢ BẰNG SIFT ====================
    sift = cv2.SIFT_create()
    diem_sift, mo_ta_sift = sift.detectAndCompute(anh_xam, None)

    print('SIFT:')
    print('So keypoint:', len(diem_sift))
    if mo_ta_sift is None:
        print('Khong co descriptors')
    else:
        print('Kich thuoc ma tran descriptors:', mo_ta_sift.shape)

    # Vẽ keypoint bằng vòng tròn đỏ nhỏ.
    anh_sift = cv2.drawKeypoints(
        anh, diem_sift, None, color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    # ==================== 4. PHÁT HIỆN VÀ MÔ TẢ BẰNG ORB ====================
    orb = cv2.ORB_create()
    diem_orb, mo_ta_orb = orb.detectAndCompute(anh_xam, None)

    print('\nORB:')
    print('So keypoint:', len(diem_orb))
    if mo_ta_orb is None:
        print('Khong co descriptors')
    else:
        print('Kich thuoc ma tran descriptors:', mo_ta_orb.shape)

    anh_orb = cv2.drawKeypoints(
        anh, diem_orb, None, color=(0, 0, 255),
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )

    # ==================== 5. GHÉP ẢNH SO SÁNH ====================
    danh_sach_anh = [anh, anh_sift, anh_orb]
    tieu_de = ['1. ANH GOC', '2. SIFT', '3. ORB']
    mau_sac = [(65, 65, 65), (0, 0, 200), (200, 75, 0)]
    anh_co_tieu_de = []

    for i in range(3):
        # Thêm lề trắng và thanh tiêu đề.
        khung = cv2.copyMakeBorder(danh_sach_anh[i], 55, 16, 16, 16,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.rectangle(khung, (0, 0), (khung.shape[1] - 1, 40), mau_sac[i], -1)
        cv2.putText(khung, tieu_de[i], (16, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Thêm viền xám ngăn cách các ảnh.
        khung = cv2.copyMakeBorder(khung, 12, 12, 12, 12,
                                  cv2.BORDER_CONSTANT, value=(220, 220, 220))
        anh_co_tieu_de.append(khung)

    anh_so_sanh = np.hstack(anh_co_tieu_de)

    # ==================== 6. LƯU ẢNH KẾT QUẢ ====================
    THU_MUC_KET_QUA.mkdir(exist_ok=True)
    ten_anh = ['anh_goc.png', 'sift_keypoints.png',
               'orb_keypoints.png', 'so_sanh_sift_orb.png']
    cac_anh = [anh, anh_sift, anh_orb, anh_so_sanh]

    for ten, hinh in zip(ten_anh, cac_anh):
        # Mã hóa PNG và lưu ảnh.
        thanh_cong, du_lieu = cv2.imencode('.png', hinh)
        if not thanh_cong:
            raise RuntimeError(f'Khong luu duoc anh {ten}')
        du_lieu.tofile(str(THU_MUC_KET_QUA / ten))

    print(f'\nDa luu anh ket qua vao: {THU_MUC_KET_QUA}')

    # ==================== 7. HIỂN THỊ KẾT QUẢ ====================
    if args.show:
        cv2.namedWindow('Cau 2 - SIFT va ORB', cv2.WINDOW_NORMAL)
        chieu_rong = min(1500, anh_so_sanh.shape[1])
        chieu_cao = round(anh_so_sanh.shape[0] * chieu_rong / anh_so_sanh.shape[1])
        cv2.resizeWindow('Cau 2 - SIFT va ORB', chieu_rong, chieu_cao)
        cv2.imshow('Cau 2 - SIFT va ORB', anh_so_sanh)
        cv2.waitKey(0)  # Nhấn một phím bất kỳ để đóng cửa sổ.
        cv2.destroyAllWindows()
