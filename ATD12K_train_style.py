import torch
import torchvision
import torchvision.transforms as TF
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from datas.AniTripletWithSGMFlow import AniTripletWithSGMFlow
from models.AnimeInterp import AnimeInterp

# Loss Functions
# Ref. https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
class StyleLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(StyleLoss, self).__init__()
        # Alphas
        self.weights = [1.0 / 2.6, 1.0 / 4.8, 1.0 / 3.7, 1.0 / 5.6, 10.0 / 1.5]

        blocks = []
        blocks.append(torchvision.models.vgg19(pretrained=True).features[:4].eval()) # relu1_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[4:9].eval()) # relu2_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[9:14].eval()) # relu3_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[18:23].eval()) # relu4_2
        blocks.append(torchvision.models.vgg19(pretrained=True).features[27:32].eval()) # relu5_2
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1))

    def vgg_loss(self, x, y):
        loss = 0.0
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y) * self.weights[i]
        return loss

    def gram_loss(self, x, y):
        loss = 0.0
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            x = block(x)
            y = block(y)
            act_x = x.reshape(x.shape[0], x.shape[1], -1)
            act_y = y.reshape(y.shape[0], y.shape[1], -1)
            gram_x = act_x @ act_x.permute(0, 2, 1)
            gram_y = act_y @ act_y.permute(0, 2, 1)
            loss += F.mse_loss(gram_x, gram_y) * self.weights[i]
        return loss

    def forward(self, output, gt):
        # Refer to FILM Section 4
        alphas = (1.0, 0.25, 40.0)

        if output.shape[1] != 3:
            output = output.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        output = (output-self.mean) / self.std
        gt = (gt-self.mean) / self.std
        if self.resize:
            output = F.interpolate(output, mode='bilinear', size=(224, 224), align_corners=False)
            gt = F.interpolate(gt, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = output
        y = gt
        
        return alphas[0] * F.l1_loss(output, gt) + alphas[1] * self.vgg_loss(x, y) + alphas[2] * self.gram_loss(x, y)

checkpoint_dir = 'checkpoints/Style/'
trainset_root = '/home/michael/hilbert/Desktop/Datasets/atd-12k/train_10k'
train_flow_root = '/home/michael/hilbert/Desktop/Datasets/atd-12k/train_10k_pre_calc_sgm_flows'
train_size = (960, 540)
train_crop_size = (380, 380)
train_batch_size = 3
inter_frames = 1
epochs = 50
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

# Loss
criterion = StyleLoss().cuda()

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
        loss = criterion(output[0], IT)
        loss.backward()
        optimizer.step()

        iLoss += loss.item()
    
    end = time.time()
    print("Epoch {} Loss: {} Time: {:.2f} min".format(epoch, iLoss, (end - start) / 60))
    with open('style-train.log', 'a') as f:
        f.write("Epoch {} Loss: {} Time: {:.2f} min\n".format(epoch, iLoss, (end - start) / 60))
    
    checkpoints = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoints, checkpoint_dir + str(epoch) + ".pth")
