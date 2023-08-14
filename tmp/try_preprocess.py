import os

import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

factors1 = [{'enhancement_factor': 1.0, 'radius': 2.0},
            {'enhancement_factor': 1.0, 'radius': 2.2},
            {'enhancement_factor': 1.2, 'radius': 2.2},
            {'enhancement_factor': 1.4, 'radius': 1.6},
            {'enhancement_factor': 1.4, 'radius': 1.8},
            {'enhancement_factor': 1.4, 'radius': 2.0},
            {'enhancement_factor': 1.6, 'radius': 1.4},
            {'enhancement_factor': 1.6, 'radius': 1.6},
            {'enhancement_factor': 1.6, 'radius': 1.8},
            {'enhancement_factor': 1.8, 'radius': 1.2},
            {'enhancement_factor': 1.8, 'radius': 2.0},
            {'enhancement_factor': 2.0, 'radius': 1.8}]

factors2 = [{'enhancement_factor': 1.4, 'radius': 2.4},
            {'enhancement_factor': 1.6, 'radius': 2.4},
            {'enhancement_factor': 1.6, 'radius': 2.8},
            {'enhancement_factor': 2.4, 'radius': 2.4},
            {'enhancement_factor': 2.6, 'radius': 2.4},
            {'enhancement_factor': 2.6, 'radius': 2.6},
            {'enhancement_factor': 2.8, 'radius': 2.6},
            {'enhancement_factor': 2.8, 'radius': 2.8}]

factors3 = [{'enhancement_factor': 1.2, 'radius': 2.2},
            {'enhancement_factor': 1.4, 'radius': 2.2},
            {'enhancement_factor': 1.4, 'radius': 2.4}]

files = []
for _, _, files in os.walk("failed_images"):
    files = files

for file in files:
    img = cv2.imread(os.path.join("failed_images", file))
    grayimage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(grayimage, cv2.MORPH_CLOSE, kernel)
    Open = cv2.morphologyEx(grayimage, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(os.path.join("failed_images", "cl" + file), closed)
    cv2.imwrite(os.path.join("failed_images", "op" + file), Open)

    closed_content = pytesseract.image_to_string(closed, lang='eng', config=r'--psm 6 outputbase digits')
    Open_content = pytesseract.image_to_string(Open, lang='eng', config=r'--psm 6 outputbase digits')

    print(file, "closed", closed_content.replace('\n', ''))
    print(file, "open", Open_content.replace('\n', ''))

    # cv2.imwrite(os.path.join("failed_images", "grayimage" + file), grayimage)
    # content = pytesseract.image_to_string(grayimage, lang='eng', config=r'--psm 6 outputbase digits')
    # print(file, "grayimage", content.replace('\n', ''))
