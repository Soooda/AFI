'''
Convert Higher vesion Pytorch weights into backward-compatiable ones
'''

import torch
import sys

print(torch.cuda.is_available())

temp = torch.load(sys.argv[1])
torch.save(temp, "output.pth", _use_new_zipfile_serialization=False)