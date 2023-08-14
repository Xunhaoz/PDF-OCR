from core import utils

utils.pdf_2_img("input_pdfs/FPFI02.pdf", "tmp/output_images/FPFI02.png")
utils.rotate_correction("tmp/output_images/FPFI02.png", "tmp/rotated_images/FPFI02.png")
utils.clip("tmp/rotated_images/FPFI02.png", "tmp/clipped_images/FPFI02.png")

# 生成好的 CSV 會在 output_csv

# 圖片格式如 FPFI07 使用 utils.extract_png_1
# 圖片格式如 FPFI02 使用 utils.extract_png_2
# 圖片格式如 FPFI03 使用 utils.extract_png_3

print("02.png")
utils.extract_png_2("tmp/clipped_images/FPFI02.png")
