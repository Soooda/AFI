import os

psnrs = {}

for checkpoint in range(160, 98, -1):
    print("Evaluating checkpoint {}.pth".format(checkpoint))

    # Change config
    with open("../configs/config_test_w_sgm.py", 'r') as f:
        txt = f.read().splitlines()
    with open("../configs/config_test_w_sgm.py", 'w') as f:
        for line in txt:
            if line.startswith("checkpoint = "):
                f.write("checkpoint = 'checkpoints/ATD12K/{}.pth'\n".format(checkpoint))
            else:
                f.write(line)
                f.write('\n')

    # Evaluate
    ret = os.system('cd ..;python test_anime_sequence_one_by_one.py configs/config_test_w_sgm.py > temp.out')

    with open("temp.out", 'r') as f:
        txt = f.read().splitlines()

    with open("atd-12k epoch{}.txt".format(checkpoint), 'w') as f:
        for line in txt[-9:]:
            f.write(line)
            f.write('\n')

        psnr = float(txt[-6][14:])
        psnrs[checkpoint] = psnr

print(psnrs)