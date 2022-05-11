'''
Check if each triplet contains valid sgm flows data
'''
import os
import numpy as np

FILE_LIST = [
    "guide_flo13.jpg",
    "guide_flo31.jpg",
    "guide_flo13.npy",
    "guide_flo31.npy",
    "matching_color_1to3.jpg",
    "matching_color_3to1.jpg",
]
sgm_folder = "../datasets/train_10k_pre_calc_sgm_flows/"

folders = os.listdir(sgm_folder)
assert len(folders) == 10000, "The total number of datasets is not 10k."

count = 0
for folder in folders:
    count += 1
    print("\rChecking {} / 10000".format(count), end="", flush="")
    ls = os.listdir(sgm_folder + folder)
    if FILE_LIST == ls:
        print(f'{folder} contains missing files.')
        continue

    try:
        np.load(sgm_folder + folder + "/guide_flo13.npy")
        np.load(sgm_folder + folder + "/guide_flo31.npy")
    except ValueError:
        print(f'{folder} contains Pickle error.')
        continue