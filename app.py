import os
import json

import cv2
import pytesseract
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path


def extract_pdf(pdf_root_path="./input_pdfs", img_root_path="./output_images"):
    pdf_names = []
    for path, names, filenames in os.walk(pdf_root_path):
        pdf_names = filenames

    # 將 pdf 轉成 png
    for pdf_name in pdf_names:
        pdf_path = os.path.join(pdf_root_path, pdf_name)
        images = convert_from_path(pdf_path, 600)
        for k, image in enumerate(images):
            img_path = os.path.join(img_root_path, f'{pdf_name[:-4]}_{k}.png')
            image.save(img_path, 'PNG')


def extract_png(img_root_path="./output_images", csv_root_path="./output_csvs", config_path="config.json"):
    with open(config_path, 'r') as file:
        data = json.load(file)

    x_table_scale = data["x_table_scale"]
    y_table_scale = data["y_table_scale"]
    x_special_scale = data["x_special_scale"]
    y_special_scale = data["y_special_scale"]

    png_names = []
    for path, names, filenames in os.walk(img_root_path):
        png_names = list(filter(lambda x: "_0.png" in x, filenames))

    # 將 png 轉成 csv
    for png_name in png_names:
        png_path = os.path.join(img_root_path, png_name)

        # 將整體 png 進行影像增強
        enhancement_factor = 2
        radius = 2
        img = Image.open(png_path)
        enhancer = ImageEnhance.Contrast(img)
        enhance_img = enhancer.enhance(enhancement_factor)
        smoothed_img = enhance_img.filter(ImageFilter.GaussianBlur(radius))

        # 分割並進行辨識 只辨識數字
        result = []
        for y_pos in y_table_scale:
            for x_pos in x_table_scale:
                cropped_img = smoothed_img.crop((x_pos[0], y_pos[0], x_pos[1], y_pos[1]))
                content = pytesseract.image_to_string(cropped_img, lang='eng', config=r'--psm 11 outputbase digits')
                result.append(content.replace("\n", ""))

        for (x_pos, y_pos) in zip(x_special_scale, y_special_scale):
            cropped_img = smoothed_img.crop((x_pos[0], y_pos[0], x_pos[1], y_pos[1]))
            content = pytesseract.image_to_string(cropped_img, lang='eng', config=r'--psm 11 outputbase digits')
            result.append(content.replace("\n", ""))

        # 資料整理成 csv 的樣子
        table = [[result[i * 6 + j] for j in range(6)] for i in range(6)]
        table.append([result[36], '', '', '', '', ''])
        table.append(['', result[37], '', result[38], '', ''])
        table.append(['', result[39], '', '', '', ''])

        table_dict = {
            'FVC': table[0],
            'FEV1': table[1],
            'FEV1/FVC': table[2],
            'MMEF 75/25': table[3],
            'PEF': table[4],
            'IC forced': table[5],
            'MVV': table[6],
            'ATS error code': table[7],
            'Level data': table[8]
        }

        index_names = ["Pred", "Pre", "%Pre/Pred", "Post", "%Post/Pred", "%Chg"]
        df = pd.DataFrame(table_dict, index=index_names).T
        df.to_csv(os.path.join(csv_root_path, png_name[:-4] + '.csv'))


def grid_visualization(ori_img_path="output_images/demo_0.png", grid_img_path="output_images/grid_visualization.png",
                       config_path="config.json"):
    color = (0, 255, 0)
    thickness = 2

    # config.json 中儲存每個格子的座標
    with open(config_path, 'r') as file:
        data = json.load(file)

    x_table_scale = data["x_table_scale"]
    y_table_scale = data["y_table_scale"]
    x_special_scale = data["x_special_scale"]
    y_special_scale = data["y_special_scale"]

    img = cv2.imread(ori_img_path)

    # 將分割的結果可視化
    for y_pos in y_table_scale:
        for x_pos in x_table_scale:
            cv2.rectangle(img, (x_pos[0], y_pos[0]), (x_pos[1], y_pos[1]), color, thickness)

    for (x_pos, y_pos) in zip(x_special_scale, y_special_scale):
        cv2.rectangle(img, (x_pos[0], y_pos[0]), (x_pos[1], y_pos[1]), color, thickness)

    cv2.imwrite(grid_img_path, img)


if __name__ == "__main__":
    # 產生 demo_0.png demo_1.png
    extract_pdf()

    # 產生 grid_visualization.png
    grid_visualization()

    # 產生 demo_0.csv
    extract_png()
