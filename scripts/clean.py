'''
Remove triplets that have been generated SGM flows to resume SGM generation
'''
import os
import shutil

names = os.listdir("../datasets/train_10k_pre_calc_sgm_flows")
print(len(names))

for name in names:
    if os.path.exists('../datasets/540p/' + name):
        shutil.rmtree('../datasets/540p/' + name)
        print("Remove Directory {}".format('../datasets/540p/' + name))