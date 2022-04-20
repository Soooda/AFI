import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
import time

from datas.QVI960 import QVI960
from models.AnimeInterp import AnimeInterp

#os.environ["CUDA_AVAILABLE_DEVICES"] = str(3)

# Loss Function
def loss_fn(output, gt):
    return F.l1_loss(output, gt)
    #return F.l1_loss(output, gt) * 204

checkpoint_dir = 'checkpoints/QVI960/'
trainset_root = 'datasets/QVI-960'
train_size = (640, 360)
train_crop_size = (352, 352)
train_batch_size = 2
epochs = 2

mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]

init_learning_rate = 1e-4
milestones = [100, 150]

# preparing transform & datasets
normalize1 = TF.Normalize(mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in mean]
revstd = [1.0/x for x in std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

trainset = QVI960(trainset_root, trans, train_size, train_crop_size, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=20)

model = AnimeInterp(path=None).cuda()
# Freeze RFR
model.flownet.requires_grad = False
# print(model)

model = nn.DataParallel(model)

# Optimizer
params = list(model.module.feat_ext.parameters()) + list(model.module.synnet.parameters())
optimizer = optim.Adam(params, lr=init_learning_rate)

# Scheduler
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

print('Everything prepared. Ready to train...')

ret = model.load_state_dict(torch.load("checkpoints/animeinterp+gma.pth")["model_state_dict"], strict=True)
print(ret)

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

start = time.time()

for epoch in range(epochs):
    iLoss = 0
    for trainIndex, (trainData, t) in enumerate(trainloader, 0):
        print(trainIndex, len(trainloader))
        # Get the input and the target from the training set
        frame0, frameT, frame1 = trainData

        I0 = frame0.cuda()
        I1 = frame1.cuda()
        IT = frameT.cuda()
        t = t.view(t.size(0,), 1, 1, 1).float().cuda()

        optimizer.zero_grad()
        output = model(I0, I1, None, None, t) # SGM flows are not used in this training
        loss = loss_fn(output[0], IT)
        loss.backward()
        optimizer.step()

        iLoss += loss.item()
    
    print(f"Epoch {epoch} Loss: {iLoss}", flush=True)
    with open('train.log', 'w') as f:
        f.write(f"Epoch {epoch} Loss: {iLoss}\n")
    
    
    torch.save(model.state_dict(), checkpoint_dir + str(epoch) + ".pth")

    # Increment scheduler count
    scheduler.step()

end = time.time()
print("Totol Running Time is {:.2f} min.".format((end - start) / 60))
with open('train.log', 'w') as f:
    f.write("Totol Running Time is {:.2f} min.\n".format((end - start) / 60))
    