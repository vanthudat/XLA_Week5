import argparse
from pathlib import Path

import cv2
import numpy as np


# ==================== 1. ĐƯỜNG DẪN HAI ẢNH ====================
THU_MUC = Path(__file__).resolve().parent
ANH_1 = THU_MUC / 'anhdung.jpg'
ANH_2 = THU_MUC / 'anhnghieng.jpg'
THU_MUC_KET_QUA = THU_MUC / 'ket_qua_cau3'


def doc_anh(duong_dan):
    # Đọc ảnh màu, hỗ trợ đường dẫn có dấu.
    if not duong_dan.is_file():
        raise FileNotFoundError(f'Khong tim thay anh: {duong_dan}')
    du_lieu = np.fromfile(str(duong_dan), dtype=np.uint8)
    anh = cv2.imdecode(du_lieu, cv2.IMREAD_COLOR)
    if anh is None:
        raise ValueError(f'Khong doc duoc anh: {duong_dan}')
    return anh


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    # ==================== 2. ĐỌC ẢNH VÀ CHUYỂN ẢNH XÁM ====================
    anh_1 = doc_anh(ANH_1)
    anh_2 = doc_anh(ANH_2)
    xam_1 = cv2.cvtColor(anh_1, cv2.COLOR_BGR2GRAY)
    xam_2 = cv2.cvtColor(anh_2, cv2.COLOR_BGR2GRAY)

    # ==================== 3. TÌM KEYPOINTS VÀ DESCRIPTORS BẰNG ORB ====================
    orb = cv2.ORB_create()
    diem_1, mo_ta_1 = orb.detectAndCompute(xam_1, None)
    diem_2, mo_ta_2 = orb.detectAndCompute(xam_2, None)

    if mo_ta_1 is None or mo_ta_2 is None:
        raise SystemExit('Khong du dac trung de doi sanh hai anh')
    if len(mo_ta_2) < 2:
        raise SystemExit('Anh thu hai can it nhat 2 descriptors de loc Ratio Test')

    # ==================== 4. ĐỐI SÁNH BẰNG BFMATCHER ====================
    # ORB có descriptor nhị phân nên dùng khoảng cách Hamming.
    bo_doi_sanh = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Mỗi descriptor của ảnh 1 tìm hai descriptor gần nhất ở ảnh 2.
    cac_cap = bo_doi_sanh.knnMatch(mo_ta_1, mo_ta_2, k=2)

    # ==================== 5. LỌC CẶP ĐỐI SÁNH BẰNG RATIO TEST ====================
    ti_le = 0.75
    cap_tot = []
    for cap in cac_cap:
        if len(cap) < 2:
            continue
        gan_nhat, gan_thu_hai = cap

        # Giữ cặp gần nhất khi khoảng cách nhỏ hơn 0.75 lần cặp thứ hai.
        if gan_nhat.distance < ti_le * gan_thu_hai.distance:
            cap_tot.append(gan_nhat)

    cap_tot.sort(key=lambda cap: cap.distance)
    print('So keypoint anh 1:', len(diem_1))
    print('So keypoint anh 2:', len(diem_2))
    print('So cap doi sanh sau Ratio Test:', len(cap_tot))

    # ==================== 6. VẼ CÁC CẶP ĐỐI SÁNH TỐT ====================
    # Nối các keypoint tương ứng giữa hai ảnh, bỏ các điểm không được ghép.
    anh_doi_sanh = cv2.drawMatches(
        anh_1, diem_1, anh_2, diem_2, cap_tot, None,
        matchColor=(0, 200, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Thêm tiêu đề cho từng ảnh.
    anh_ket_qua = cv2.copyMakeBorder(
        anh_doi_sanh, 50, 12, 12, 12,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(anh_ket_qua, '1. ANH DUNG', (24, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(anh_ket_qua, '2. ANH NGHIENG', (anh_1.shape[1] + 24, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # ==================== 7. LƯU ẢNH KẾT QUẢ ====================
    THU_MUC_KET_QUA.mkdir(exist_ok=True)
    duong_dan_luu = THU_MUC_KET_QUA / 'orb_doi_sanh.png'
    thanh_cong, du_lieu = cv2.imencode('.png', anh_ket_qua)
    if not thanh_cong:
        raise RuntimeError('Khong luu duoc anh doi sanh')
    du_lieu.tofile(str(duong_dan_luu))
    print(f'Da luu anh ket qua vao: {THU_MUC_KET_QUA}')

    # ==================== 8. HIỂN THỊ KẾT QUẢ ====================
    if args.show:
        cv2.namedWindow('Cau 3 - Doi sanh ORB', cv2.WINDOW_NORMAL)
        chieu_rong = min(1500, anh_ket_qua.shape[1])
        chieu_cao = round(anh_ket_qua.shape[0] * chieu_rong / anh_ket_qua.shape[1])
        cv2.resizeWindow('Cau 3 - Doi sanh ORB', chieu_rong, chieu_cao)
        cv2.imshow('Cau 3 - Doi sanh ORB', anh_ket_qua)
        cv2.waitKey(0)  # Nhấn một phím bất kỳ để đóng cửa sổ.
        cv2.destroyAllWindows()
