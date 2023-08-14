import os
import json

from score import score

import cv2
import PIL.Image
import pdf2image
import pytesseract
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter


def rotate_correction_visualize():
    img = cv2.imread('../tmp/output_images/FPFI07.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    print(coords)

    rect = cv2.minAreaRect(coords)
    rect = (rect[0], rect[1], 180 - rect[2])
    print(rect[2])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(gray, [box], 0, (0, 255, 0), 10)
    cv2.imshow('gray bound', cv2.resize(gray, (1000, 1500)))

    angle = cv2.minAreaRect(coords)[-1]
    print(angle)

    angle = 90 - abs(angle)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    print(angle)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.putText(rotated, 'Angle: {:.2f} degrees'.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    print('[INFO] angel :{:.3f}'.format(angle))
    cv2.imshow('Input', cv2.resize(img, (1000, 1500)))
    cv2.imshow('Rotated', cv2.resize(rotated, (1000, 1500)))


def pdf_2_img_batch(source='input_pdfs', dest='tmp/output_images'):
    filenames = []
    for _, _, filenames in os.walk(source):
        filenames = list(filter(lambda x: '.pdf' in x, filenames))

    for filename in tqdm(filenames):
        pages = pdf2image.convert_from_path(
            os.path.join(source, filename),
            dpi=600
        )
        pages[0].save(os.path.join(dest, filename[:-4] + '.png'))


def pdf_2_img(source, dest):
    pages = pdf2image.convert_from_path(
        source,
        dpi=600
    )
    pages[0].save(os.path.join(dest))


def rotate_correction_batch(source='tmp/output_images', dest='tmp/rotated_images'):
    filenames = []
    for path, names, filenames in os.walk(source):
        filenames = list(filter(lambda x: ".png" in x, filenames))

    for filename in tqdm(filenames):
        source_file = os.path.join(source, filename)
        dest_file = os.path.join(dest, filename)

        img = cv2.imread(source_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        angle = 90 - abs(angle)

        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(dest_file, rotated)


def rotate_correction(source, dest):
    img = cv2.imread(source)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = 90 - abs(angle)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imwrite(dest, rotated)


def clip_batch(source='tmp/rotated_images', dest='tmp/clipped_images'):
    filenames = []
    for path, names, filenames in os.walk(source):
        filenames = list(filter(lambda x: ".png" in x, filenames))

    for filename in tqdm(filenames):
        source_file = os.path.join(source, filename)
        dest_file = os.path.join(dest, filename)

        img = cv2.imread(source_file)
        gray = cv2.imread(source_file, cv2.IMREAD_GRAYSCALE)
        gray = gray[500:6000, :]
        ret, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        x_projection = np.max(binary_img, axis=0)
        y_projection = np.max(binary_img, axis=1)

        start_x = np.argmax(x_projection > 0)
        end_x = len(x_projection) - 1 - np.argmax(np.flip(x_projection) > 0)
        start_y = np.argmax(y_projection > 0)
        end_y = len(y_projection) - 1 - np.argmax(np.flip(y_projection) > 0)

        img = img[500 + start_y: 500 + end_y, start_x: end_x]
        cv2.imwrite(dest_file, img)


def clip(source, dest):
    img = cv2.imread(source)
    gray = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    gray = gray[500:6000, :]
    ret, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    x_projection = np.max(binary_img, axis=0)
    y_projection = np.max(binary_img, axis=1)

    start_x = np.argmax(x_projection > 0)
    end_x = len(x_projection) - 1 - np.argmax(np.flip(x_projection) > 0)
    start_y = np.argmax(y_projection > 0)
    end_y = len(y_projection) - 1 - np.argmax(np.flip(y_projection) > 0)

    img = img[500 + start_y: 500 + end_y, start_x: end_x]
    cv2.imwrite(dest, img)


def grid_visualization_1(ori_img_path=""):
    color = (0, 0, 0)
    thickness = 10
    img = cv2.imread(ori_img_path)
    img = cv2.resize(img, (3778, 3147))

    X = [1050, 1350, 1750, 2200, 2600, 3100, 3550]
    Y = [740, 840, 940, 1040, 1150, 1250, 1350, 1460, 1570]
    date_pt = [[1350, 1570], [1900, 1700]]

    for x_i, x in enumerate(X[:-1]):
        for y_i, y in enumerate(Y[:-1]):
            cv2.rectangle(img, (x, y), (X[x_i + 1], Y[y_i + 1]), color, thickness)

    cv2.rectangle(img, *date_pt, color, thickness)
    cv2.imshow(f'{os.path.basename(ori_img_path)}', cv2.resize(img, (500, 750)))


def grid_visualization_2(ori_img_path=""):
    color = (0, 0, 0)
    thickness = 10
    img = cv2.imread(ori_img_path)
    img = cv2.resize(img, (3778, 3147))

    X = [1030, 1300, 1750, 2100, 2600, 3100, 3550]
    Y = [820, 910, 1000, 1110, 1210, 1310, 1410, 1510, 1610]
    date_pt = [[1030, 1610], [1300, 1720]]

    for x_i, x in enumerate(X[:-1]):
        for y_i, y in enumerate(Y[:-1]):
            cv2.rectangle(img, (x, y), (X[x_i + 1], Y[y_i + 1]), color, thickness)

    cv2.rectangle(img, *date_pt, color, thickness)
    cv2.imshow(f'{os.path.basename(ori_img_path)}', cv2.resize(img, (500, 750)))


def grid_visualization_3(ori_img_path=""):
    color = (0, 0, 0)
    thickness = 10
    img = cv2.imread(ori_img_path)
    img = cv2.resize(img, (3778, 3147))

    X = [1030, 1300, 1750, 2100, 2600, 3100, 3550]
    Y = [890, 1000, 1120, 1230, 1350, 1450, 1570, 1680, 1790]

    for x_i, x in enumerate(X[:-1]):
        for y_i, y in enumerate(Y[:-1]):
            cv2.rectangle(img, (x, y), (X[x_i + 1], Y[y_i + 1]), color, thickness)
    cv2.imshow(f'{os.path.basename(ori_img_path)}', cv2.resize(img, (500, 750)))


def voted_extract(img: PIL.Image.Image, factors: list, bound_100: bool):
    vote_dict = {}

    for factor in factors:
        enhancement_factor = factor['enhancement_factor']
        radius = factor['radius']

        tmp_img = img.copy()
        smoothed_img = tmp_img.filter(ImageFilter.GaussianBlur(radius))

        enhancer = ImageEnhance.Contrast(smoothed_img)
        enhance_img = enhancer.enhance(enhancement_factor)

        content = pytesseract.image_to_string(enhance_img, lang='eng', config=r'--psm 6 outputbase digits')
        content = content.replace('\n', '').replace(' ', '')

        if content not in vote_dict:
            vote_dict[content] = 0
        else:
            vote_dict[content] += 1

    # print(vote_dict)
    max_key = max(vote_dict, key=lambda k: vote_dict[k])
    if max_key == "":
        return None

    if max_key == ".":
        return None

    if "." in max_key or "-" in max_key:
        return max_key

    if float(max_key) > 100 and bound_100:
        return float(max_key) / 100

    return float(max_key)


def extract_png_1(img_path="", config_path="configs/clipped_grid_1.json"):
    with open(config_path, 'r') as file:
        data = json.load(file)

    file_name = os.path.basename(img_path)
    X = data["X"]
    Y = data["Y"]
    date_pt = data["date_pt"]

    factors = [{'enhancement_factor': 1.4, 'radius': 2.4},
               {'enhancement_factor': 1.6, 'radius': 2.4},
               {'enhancement_factor': 1.6, 'radius': 2.8},
               {'enhancement_factor': 2.4, 'radius': 2.4},
               {'enhancement_factor': 2.6, 'radius': 2.4},
               {'enhancement_factor': 2.6, 'radius': 2.6},
               {'enhancement_factor': 2.8, 'radius': 2.6},
               {'enhancement_factor': 2.8, 'radius': 2.8}]
    img = Image.open(img_path)
    img = img.resize((3778, 3147))

    # 分割並進行辨識 只辨識數字
    result = []
    for x_i, x in enumerate(X[:-1]):
        for y_i, y in enumerate(Y[:-1]):
            cropped_img = img.crop((x, y, X[x_i + 1], Y[y_i + 1]))
            cropped_img.save(f'tmp/failed_images/{file_name.split(".")[0]}_{x_i * len(Y) + y_i}' + '.png')
            content = voted_extract(cropped_img, factors=factors, bound_100=False)
            result.append(content)

    cropped_img = img.crop((date_pt[0][0], date_pt[0][1], date_pt[1][0], date_pt[1][1]))
    date_time = voted_extract(cropped_img, factors=factors, bound_100=False)

    result = [[result[i * 8 + j] for j in range(8)] for i in range(6)]
    result[1].append(date_time)

    for i in range(6):
        if i > 0:
            result[i][5] = None
        if i == 0 or i == 2 or i > 3:
            result[i][7] = None

    df = pd.DataFrame(result).T
    df.to_csv(f"output_csvs/{file_name.split('.')[0]}.csv", index=False, header=False)
    score.score(f"score/ac_csvs/{file_name.split('.')[0]}.csv", f"output_csvs/{file_name.split('.')[0]}.csv")


def extract_png_2(img_path, config_path="configs/clipped_grid_2.json"):
    with open(config_path, 'r') as file:
        data = json.load(file)

    file_name = os.path.basename(img_path)
    X = data["X"]
    Y = data["Y"]
    ic_forced = data["ic_forced"]

    factors = [{'enhancement_factor': 1.0, 'radius': 2.0},
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

    img = Image.open(img_path)
    img = img.resize((3778, 3147))

    # 分割並進行辨識 只辨識數字
    result = []
    for x_i, x in enumerate(X[:-1]):
        for y_i, y in enumerate(Y[:-1]):
            cropped_img = img.crop((x, y, X[x_i + 1], Y[y_i + 1]))
            cropped_img.save(f'tmp/failed_images/{file_name.split(".")[0]}_{x_i * len(Y) + y_i}' + '.png')
            content = voted_extract(cropped_img, factors, True)
            result.append(content)

    cropped_img = img.crop((ic_forced[0][0], ic_forced[0][1], ic_forced[1][0], ic_forced[1][1]))
    ic_forced_value = voted_extract(cropped_img, factors, True)

    result = [[result[i * 8 + j] for j in range(8)] for i in range(6)]
    result[0].append(ic_forced_value)

    for i in range(6):
        if i == 0 or i == 2 or i == 4:
            result[i][3] = None
            result[i][4] = None
            result[i][7] = None

        if i != 0:
            result[i].append(None)

    df = pd.DataFrame(result).T
    df.to_csv(f"output_csvs/{file_name.split('.')[0]}.csv", index=False, header=False)
    score.score(f"score/ac_csvs/{file_name.split('.')[0]}.csv", f"output_csvs/{file_name.split('.')[0]}.csv")


def extract_png_3(img_path="", config_path="configs/clipped_grid_3.json"):
    with open(config_path, 'r') as file:
        data = json.load(file)

    file_name = os.path.basename(img_path)
    X = data["X"]
    Y = data["Y"]

    factors = [{'enhancement_factor': 1.2, 'radius': 2.2},
               {'enhancement_factor': 1.4, 'radius': 2.2},
               {'enhancement_factor': 1.4, 'radius': 2.4}]

    img = Image.open(img_path)
    img = img.resize((3778, 3147))

    # 分割並進行辨識 只辨識數字
    result = []

    for x_i, x in enumerate(X[:-1]):
        for y_i, y in enumerate(Y[:-1]):
            cropped_img = img.crop((x, y, X[x_i + 1], Y[y_i + 1]))
            cropped_img.save(f'tmp/failed_images/{file_name.split(".")[0]}_{x_i * len(Y) + y_i}' + '.png')
            content = voted_extract(cropped_img, factors, True)
            result.append(content)

    result = [[result[i * 8 + j] for j in range(8)] for i in range(6)]

    for i in range(6):
        if i == 0 or i == 2 or i == 4:
            result[i][3] = None
            result[i][4] = None
            result[i][7] = None

    df = pd.DataFrame(result).T
    df.to_csv(f"output_csvs/{file_name.split('.')[0]}.csv", index=False, header=False)
    score.score(f"score/ac_csvs/{file_name.split('.')[0]}.csv", f"output_csvs/{file_name.split('.')[0]}.csv")

# rotate_correction_visualize()
# grid_visualization("clipped_images/FPFI06.png")
# grid_visualization_1("clipped_images/demo.png")
# grid_visualization_2("clipped_images/FPFI02.png")
# grid_visualization_3("clipped_images/FPFI03.png")
# cv2.waitKey()

# pdf_2_img_batch()
# rotate_correction_batch()
# clip_batch()

# print("07.png")
# extract_png_1("tmp/clipped_images/FPFI07.png")
# print("06.png")
# extract_png_1("tmp/clipped_images/FPFI06.png")
# print("05.png")
# extract_png_1("tmp/clipped_images/FPFI05.png")
# print("04.png")
# extract_png_1("tmp/clipped_images/FPFI04.png")
# print("02.png")
# extract_png_2("tmp/clipped_images/FPFI02.png")
# print("03.png")
# extract_png_3("../tmp/clipped_images/FPFI03.png")
