import os
import json

from score.score import score

import cv2
import pytesseract
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path


def df_save(df, name):
    df.iloc[[6, 5], :] = df.iloc[[5, 6], :]
    df.index = ['FVC(L)', 'FEV1(L)', 'FEV1/FVC(%)', 'MMEF 75/25(L/s)', 'PEF(L/s)', 'MVV(L/min)', 'IC forced(L)',
                'ATS error code', 'Level data']
    df.columns = ['Pred', 'Pre', '%Pre/Pred', 'Post', '%Post/Pred', '%Chg']
    df.to_csv(f'output_csvs/{name}.csv')


def df_check(result):
    table = [[result[i * 6 + j] for j in range(6)] for i in range(6)]
    table.append([result[36], '', '', '', '', ''])
    table.append(['', result[37], '', result[38], '', ''])
    table.append(['', result[39], '', '', '', ''])

    df = pd.DataFrame(table)
    df.reset_index(drop=True, inplace=True)
    df.to_csv("score\check.csv", index=False, header=False)
    return df


def extract_pdf(pdf_root_path="./input_pdfs", img_root_path="./output_images"):
    pdf_names = []
    for path, names, filenames in os.walk(pdf_root_path):
        pdf_names = filter(lambda x: ".pdf" in x, filenames)

    # 將 pdf 轉成 png
    for pdf_name in pdf_names:
        pdf_path = os.path.join(pdf_root_path, pdf_name)
        images = convert_from_path(pdf_path, 600)
        img_path = os.path.join(img_root_path, f'{pdf_name[:-4]}.png')
        images[0].save(img_path, 'PNG')


def extract_png(img_root_path="./output_images", config_path="config.json"):
    with open(config_path, 'r') as file:
        data = json.load(file)

    x_table_scale = data["x_table_scale"]
    y_table_scale = data["y_table_scale"]
    x_special_scale = data["x_special_scale"]
    y_special_scale = data["y_special_scale"]

    png_names = []
    for path, names, filenames in os.walk(img_root_path):
        png_names = list(filter(lambda x: ".png" in x, filenames))

    # 將 png 轉成 csv
    for png_name in png_names:
        png_path = os.path.join(img_root_path, png_name)

        # 將整體 png 進行影像增強
        enhancement_factor = 2
        radius = 2.6
        img = Image.open(png_path)
        smoothed_img = img.filter(ImageFilter.GaussianBlur(radius))
        enhancer = ImageEnhance.Contrast(smoothed_img)
        enhance_img = enhancer.enhance(enhancement_factor)

        enhance_img.save(f"processed_images/{png_name}")

        # 分割並進行辨識 只辨識數字
        result = []
        cnt = 0
        for y_pos in y_table_scale:
            for x_pos in x_table_scale:
                cropped_img = enhance_img.crop((x_pos[0], y_pos[0], x_pos[1], y_pos[1]))
                cnt += 1
                content = pytesseract.image_to_string(cropped_img, lang='eng',
                                                      config=r'--psm 6outputbase digits')
                result.append(content.replace("\n", ""))

        for (x_pos, y_pos) in zip(x_special_scale, y_special_scale):
            cropped_img = enhance_img.crop((x_pos[0], y_pos[0], x_pos[1], y_pos[1]))
            cnt += 1
            content = pytesseract.image_to_string(cropped_img, lang='eng', config=r'--psm 6 outputbase digits')
            result.append(content.replace("\n", ""))

        df = df_check(result)
        df_save(df, png_name[:-4])


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
    # pip line
    extract_pdf()
    extract_png()
    score()
