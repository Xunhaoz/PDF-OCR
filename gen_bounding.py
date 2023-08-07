import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片並轉換為灰度圖
image_path = 'output_images/FPFI07.png'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用HSV色彩空間來識別綠色
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
mask = cv2.inRange(hsv, lower_green, upper_green)

# 使用Canny邊緣檢測找到圖片中的邊緣
edges = cv2.Canny(mask, 50, 150, apertureSize=3)

# 使用Hough變換檢測直線
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# 找到最長的綠色直線
max_length = 0
longest_line = None
for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length > max_length:
        max_length = length
        longest_line = line

print(longest_line)

# 繪製最長的綠色直線
line_image = image.copy()
x1, y1, x2, y2 = longest_line[0]
cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 0), 4)

# 可視化原始圖像和含有最長綠色直線的圖像
fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=600)

# 顯示原始圖像
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title('Original Image')

# 顯示含有最長綠色直線的圖像
axs[1].imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
axs[1].set_title('Image with Longest Green Line')

for ax in axs:
    ax.axis('off')

plt.show()
