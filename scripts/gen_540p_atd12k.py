import cv2
import os

input_dir = "datasets/train_10k/"
output_dir = "datasets/train_10k_540p/"
resize_dim = (960, 540)

folders = os.listdir(input_dir)

os.mkdir(output_dir)

for i, cur in enumerate(folders):
    files = os.listdir(input_dir + cur)
    os.mkdir(output_dir + cur)
    for f in files:
        fname, _ = f.split(".")
        img = cv2.imread(input_dir + cur + "/" + f, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, resize_dim, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_dir + cur + "/" + fname + ".jpg", resized)
    print("Progress: {:.2f}%".format((i + 1) / len(folders)))