import torch
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from datas.AniTripletWithSGMFlow import AniTripletWithSGMFlow
from models.AnimeInterp import AnimeInterp

# Loss Functions
def loss_fn(output, gt):
    return F.l1_loss(output, gt)

checkpoint_dir = 'checkpoints/ATD12K/'
trainset_root = 'datasets/atd-12k/train_10k'
train_flow_root = 'datasets/atd-12k/train_10k_pre_calc_sgm_flows'
train_size = (960, 540)
train_crop_size = (380, 380)
train_batch_size = 3
inter_frames = 1
epochs = 100
init_learning_rate = 1e-6

mean = [0., 0., 0.]
std  = [1, 1, 1]

normalize1 = TF.Normalize(mean, [1.0, 1.0, 1.0])
normalize2 = TF.Normalize([0, 0, 0], std)
trans = TF.Compose([TF.ToTensor(), normalize1, normalize2, ])

revmean = [-x for x in mean]
revstd = [1.0 / x for x in std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

revtrans = TF.Compose([revnormalize1, revnormalize2, TF.ToPILImage()])

trainset = AniTripletWithSGMFlow(trainset_root, train_flow_root, trans, train_size, train_crop_size)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=20)

model = AnimeInterp(path=None).cuda()
model = nn.DataParallel(model)

# Optimizer
params = model.parameters()
optimizer = optim.Adam(params, lr=init_learning_rate)

ret = model.load_state_dict(torch.load("checkpoints/QVI960/199.pth")["state_dict"], strict=True)
print(ret)

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

print('Everything prepared. Ready to train...')

for epoch in range(epochs):
    # Resume
    if os.path.exists(checkpoint_dir + str(epoch) + ".pth"):
        temp = torch.load(checkpoint_dir + str(epoch) + ".pth")
        ret = model.load_state_dict(temp['state_dict'])
        print(ret)
        optimizer.load_state_dict(temp['optimizer'])

        # optimizer.param_groups[0]['lr'] = 0.000001

        print(f"Epoch {epoch} Checkpoint loaded!")
        continue

    iLoss = 0
    start = time.time()
    for trainIndex, trainData in enumerate(trainloader, 0):
        print(f"\r{trainIndex+1}/{len(trainloader)}", end='', flush='')

        # Get the input and the target from the training set
        sample, flow = trainData
        frame1 = sample[0]
        frame2 = sample[-1]
        frameT = sample[1]

        # initial SGM flow
        F12i, F21i  = flow
        F12i = F12i.float().cuda() 
        F21i = F21i.float().cuda()

        I0 = frame1.cuda()
        I1 = frame2.cuda()
        IT = frameT.cuda()
        t = 0.5

        optimizer.zero_grad()
        output = model(I0, I1, F12i, F21i, t)
        loss = loss_fn(output[0], IT)
        loss.backward()
        optimizer.step()

        iLoss += loss.item()
    
    end = time.time()
    print("Epoch {} Loss: {} Time: {:.2f} min".format(epoch, iLoss, (end - start) / 60))
    with open('train.log', 'a') as f:
        f.write("Epoch {} Loss: {} Time: {:.2f} min\n".format(epoch, iLoss, (end - start) / 60))
    
    checkpoints = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoints, checkpoint_dir + str(epoch) + ".pth")
