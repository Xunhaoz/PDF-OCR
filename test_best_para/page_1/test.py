import json
import os
import score
import pandas as pd

import pytesseract

import PIL
from PIL import ImageFilter, ImageEnhance, Image


def voted_extract_test(img: PIL.Image.Image, factors: list, bound_100: bool):
    tmp_img = img.copy()
    enhance_img = tmp_img.filter(ImageFilter.GaussianBlur(factors[0]))

    enhancer = ImageEnhance.Brightness(enhance_img)
    enhance_img = enhancer.enhance(factors[1])

    enhancer = ImageEnhance.Contrast(enhance_img)
    enhance_img = enhancer.enhance(factors[2])

    enhancer = ImageEnhance.Color(enhance_img)
    enhance_img = enhancer.enhance(factors[3])

    enhancer = ImageEnhance.Sharpness(enhance_img)
    enhance_img = enhancer.enhance(factors[4])

    content = pytesseract.image_to_string(enhance_img, lang='eng', config=r'--psm 6 outputbase digits')
    content = content.replace('\n', '').replace(' ', '')

    if content == "":
        return None

    if content == ".":
        return None

    if "." in content or "-" in content:
        return content

    if float(content) > 100 and bound_100:
        return float(content) / 100

    return float(content)


def extract_png_test_1(img_path="", config_path="clipped_grid_1.json"):
    with open(config_path, 'r') as file:
        data = json.load(file)

    file_name = os.path.basename(img_path)
    X = data["X"]
    Y = data["Y"]
    date_pt = data["date_pt"]

    for factor_i in range(10, 61, 5):
        for factor_j in range(10, 21, 2):
            for factor_k in range(10, 21, 2):
                for factor_l in range(10, 21, 2):
                    for factor_m in range(2, 9, 1):

                        factors = [factor_i / 10, factor_j / 10, factor_k / 10, factor_l / 10, factor_m / 10]

                        with open("page_1.txt", "a") as f:
                            f.write(f"factor: {factors}\n")

                        print(f"factor: {factors}")

                        img = Image.open(img_path)
                        img = img.resize((3778, 3147))

                        # 分割並進行辨識 只辨識數字
                        result = []
                        for x_i, x in enumerate(X[:-1]):
                            for y_i, y in enumerate(Y[:-1]):
                                cropped_img = img.crop((x, y, X[x_i + 1], Y[y_i + 1]))
                                content = voted_extract_test(cropped_img, factors=factors, bound_100=False)
                                result.append(content)

                        cropped_img = img.crop((date_pt[0][0], date_pt[0][1], date_pt[1][0], date_pt[1][1]))
                        date_time = voted_extract_test(cropped_img, factors=factors, bound_100=False)

                        result = [[result[i * 8 + j] for j in range(8)] for i in range(6)]
                        result[1].append(date_time)

                        for i in range(6):
                            if i > 0:
                                result[i][5] = None
                            if i == 0 or i == 2 or i > 3:
                                result[i][7] = None

                        df = pd.DataFrame(result).T
                        df.to_csv(f"{file_name.split('.')[0]}.csv", index=False, header=False)
                        score.score(f"{file_name.split('.')[0]}.csv", f"ac_{file_name.split('.')[0]}.csv")


extract_png_test_1("FPFI05.png")
