import cv2
import numpy as np

ERROR = 1234


# 度數轉換
def degree_trans(theta):
    return theta / np.pi * 180


# 逆時針旋轉圖像degree角度（原尺寸）
def rotate_image(src, degree):
    center = (src.shape[1] // 2, src.shape[0] // 2)
    length = int(np.sqrt(src.shape[1] ** 2 + src.shape[0] ** 2))
    M = cv2.getRotationMatrix2D(center, degree, 1)
    rotated_image = cv2.warpAffine(src, M, (length, length), borderValue=(255, 255, 255))
    return rotated_image


# 通過霍夫變換計算角度
def calc_degree(src_image):
    mid_image = cv2.Canny(src_image, 50, 200, 3)
    dst_image = cv2.cvtColor(mid_image, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(mid_image, 1, np.pi / 180, 300, 0, 0)

    if not lines is None and len(lines) == 0:
        lines = cv2.HoughLines(mid_image, 1, np.pi / 180, 200, 0, 0)

    if not lines is None and len(lines) == 0:
        lines = cv2.HoughLines(mid_image, 1, np.pi / 180, 150, 0, 0)

    if lines is None or len(lines) == 0:
        print("沒有檢測到直線！")
        return ERROR

    sum_theta = 0
    for line in lines:
        rho, theta = line[0]
        sum_theta += theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
        cv2.line(dst_image, pt1, pt2, (55, 100, 195), 1, cv2.LINE_AA)

    average_theta = sum_theta / len(lines)
    print("average theta:", average_theta)

    angle = degree_trans(average_theta) - 90
    return angle


def image_recify(input_file, output_file):
    degree = 0
    src = cv2.imread(input_file)
    cv2.imshow("ori", cv2.resize(src, (1000, 1400)))

    # 傾斜角度校正
    degree = calc_degree(src)
    if degree == ERROR:
        print("校正失敗！")
        return

    rotated_dst = rotate_image(src, degree)
    print("angle:", degree)
    cv2.imshow("aft", cv2.resize(rotated_dst, (1000, 1400)))

    result_image = rotated_dst[0:500, 0:rotated_dst.shape[1]]  # 根據先驗知識，估計好文本的長寬，再裁剪下來
    cv2.imshow("fin", cv2.resize(result_image, (1000, 1400)))
    cv2.imwrite(output_file, result_image)


if __name__ == "__main__":
    image_recify("output_images/test.png", "FinalImage.jpg")
    cv2.waitKey()
    cv2.destroyAllWindows()
