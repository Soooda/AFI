from tqdm import tqdm
import cv2
import os
import skvideo.io
import torch
import torchvision.transforms as TF
import numpy as np
from queue import Queue
import _thread
import warnings

warnings.filterwarnings("ignore")

from models import AnimeInterp
from models.SGM.my_models import create_VGGFeatNet
from models.SGM.gen_sgm import dline_of, trapped_ball_processed, squeeze_label_map, superpixel_pooling, superpixel_count, scatter_mean, get_guidance_flow, mutual_matching

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.set_default_tensor_type(torch.cuda.HalfTensor)
else:
    device = torch.device("cpu")

vggNet = create_VGGFeatNet().to(device)
toTensor = TF.ToTensor()
normalize = TF.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])


mean = [0., 0., 0.]
std  = [1, 1, 1]
revmean = [-x for x in mean]
revstd = [1.0 / x for x in std]
revnormalize1 = TF.Normalize([0.0, 0.0, 0.0], revstd)
revnormalize2 = TF.Normalize(revmean, [1.0, 1.0, 1.0])
revNormalize = TF.Compose([revnormalize1, revnormalize2])

def generate_sgm(I1, I3):
    use_gpu = True
    rankSumThr = 0
    # Japan
    boundImg1 = dline_of(I1, 2, 20, [10,10,10]).astype(np.uint8)
    boundImg3 = dline_of(I3, 2, 20, [10,10,10]).astype(np.uint8)
    # Disney
    # boundImg1 = dline_of(I1, 1, 20, [30,40,30]).astype(np.uint8)
    # boundImg3 = dline_of(I3, 1, 20, [30,40,30]).astype(np.uint8)
    ret, binMap1 = cv2.threshold(boundImg1, 220, 255, cv2.THRESH_BINARY)
    ret, binMap3 = cv2.threshold(boundImg3, 220, 255, cv2.THRESH_BINARY)

    # Trapped Ball Processed
    fillMap1 = trapped_ball_processed(binMap1, I1)
    fillMap3 = trapped_ball_processed(binMap3, I3)

    labelMap1 = squeeze_label_map(fillMap1)
    labelMap3 = squeeze_label_map(fillMap3)

    # VGG features
    img1_rgb = cv2.cvtColor(I1, cv2.COLOR_BGR2RGB)
    img3_rgb = cv2.cvtColor(I3, cv2.COLOR_BGR2RGB)

    img1_tensor = normalize(toTensor(img1_rgb/255.).float())
    img1_tensor = img1_tensor.unsqueeze(dim=0)
    img3_tensor = normalize(toTensor(img3_rgb/255.).float())
    img3_tensor = img3_tensor.unsqueeze(dim=0)
    if use_gpu:
        img1_tensor = img1_tensor.cuda()
        img3_tensor = img3_tensor.cuda()

    featx1_1, featx2_1, featx4_1, featx8_1, featx16_1 = vggNet(img1_tensor)
    featx1_3, featx2_3, featx4_3, featx8_3, featx16_3 = vggNet(img3_tensor)

    # Compute Correlation Map
    # superpixel pooling
    labelMap1_x2 = labelMap1[1::2,1::2]
    labelMap1_x4 = labelMap1_x2[1::2,1::2]
    labelMap1_x8 = labelMap1_x4[1::2,1::2]
    # labelMap1_x16 = labelMap1_x8[1::2,1::2]
    labelMap3_x2 = labelMap3[1::2,1::2]
    labelMap3_x4 = labelMap3_x2[1::2,1::2]
    labelMap3_x8 = labelMap3_x4[1::2,1::2]
    # labelMap3_x16 = labelMap3_x8[1::2,1::2]

    featx1_pool_1 = superpixel_pooling(featx1_1[0], labelMap1, use_gpu)
    featx2_pool_1 = superpixel_pooling(featx2_1[0], labelMap1_x2, use_gpu)
    featx4_pool_1 = superpixel_pooling(featx4_1[0], labelMap1_x4, use_gpu)
    featx8_pool_1 = superpixel_pooling(featx8_1[0], labelMap1_x8, use_gpu)
    # featx16_pool_1 = superpixel_pooling(featx16_1[0], labelMap1_x16, use_gpu)
    featx1_pool_3 = superpixel_pooling(featx1_3[0], labelMap3, use_gpu)
    featx2_pool_3 = superpixel_pooling(featx2_3[0], labelMap3_x2, use_gpu)
    featx4_pool_3 = superpixel_pooling(featx4_3[0], labelMap3_x4, use_gpu)
    featx8_pool_3 = superpixel_pooling(featx8_3[0], labelMap3_x8, use_gpu)
    # featx16_pool_3 = superpixel_pooling(featx16_3[0], labelMap3_x16, use_gpu)
    
    feat_pool_1 = torch.cat([featx1_pool_1, featx2_pool_1, featx4_pool_1, featx8_pool_1], dim=0)
    feat_pool_3 = torch.cat([featx1_pool_3, featx2_pool_3, featx4_pool_3, featx8_pool_3], dim=0)

    # normalization
    feat_p1_tmp = feat_pool_1 - feat_pool_1.min(dim=0)[0]
    feat_p1_norm = feat_p1_tmp/feat_p1_tmp.sum(dim=0)
    feat_p3_tmp = feat_pool_3 - feat_pool_3.min(dim=0)[0]
    feat_p3_norm = feat_p3_tmp/feat_p3_tmp.sum(dim=0)


    # for pixel distance
    lH, lW = labelMap1.shape
    gridX, gridY = np.meshgrid(np.arange(lW), np.arange(lH))

    gridX_flat = torch.tensor(gridX.astype(np.float), requires_grad=False).reshape(lH*lW)
    gridY_flat = torch.tensor(gridY.astype(np.float), requires_grad=False).reshape(lH*lW)

    labelMap1_flat = torch.tensor(labelMap1.reshape(lH*lW)).long()
    labelMap3_flat = torch.tensor(labelMap3.reshape(lH*lW)).long()
    
    if use_gpu:
        gridX_flat = gridX_flat.cuda()
        gridY_flat = gridY_flat.cuda()
        labelMap1_flat = labelMap1_flat.cuda()
        labelMap3_flat = labelMap3_flat.cuda()

    mean_X_1 = scatter_mean(gridX_flat, labelMap1_flat).cpu().numpy()
    mean_Y_1 = scatter_mean(gridY_flat, labelMap1_flat).cpu().numpy()
    mean_X_3 = scatter_mean(gridX_flat, labelMap3_flat).cpu().numpy()
    mean_Y_3 = scatter_mean(gridY_flat, labelMap3_flat).cpu().numpy()

    # pixel count in superpixel
    pixelCounts_1 = superpixel_count(labelMap1)
    pixelCounts_3 = superpixel_count(labelMap3)

    # some other distance
    labelNum_1 = len(np.unique(labelMap1))
    labelNum_3 = len(np.unique(labelMap3))
    # print('label num: %d, %d'%(labelNum_1, labelNum_3))

    maxDist = np.linalg.norm([lH,lW])
    maxPixNum = lH*lW

    corrMap = torch.zeros(labelNum_1, labelNum_3)
    ctxSimMap = torch.zeros(labelNum_1, labelNum_3)

    for x in range(labelNum_1):
        for y in range(labelNum_3):
            corrMap[x,y] = torch.sum(torch.min(feat_p1_norm[:,x], feat_p3_norm[:,y]))

            # pixel number as similarity
            num_1 = float(pixelCounts_1[x])
            num_3 = float(pixelCounts_3[y])
            
            sizeDiff = max(num_1/num_3, num_3/num_1)
            if sizeDiff > 3:
                corrMap[x,y] -= sizeDiff/20

            # spatial distance as similarity
            dist = np.linalg.norm([mean_X_1[x] - mean_X_3[y], mean_Y_1[x] - mean_Y_3[y]])/maxDist
            
            if dist > 0.14:
                corrMap[x,y] -= dist/5


    matchingMetaData = mutual_matching(corrMap)
    rankSum_1to3, matching_1to3, sortedCorrMap_1 = matchingMetaData[:3]
    rankSum_3to1, matching_3to1, sortedCorrMap_3 = matchingMetaData[3:]

    mMatchCount_1 = (rankSum_1to3 <= rankSumThr).sum()
    mMatchCount_3 = (rankSum_3to1 <= rankSumThr).sum()
    totalMatchCount += (mMatchCount_1 + mMatchCount_3)/2

    # Generating flows
    guideflow_1to3, matching_color_1to3 = get_guidance_flow(labelMap1, labelMap3, I1, I3,
                            rankSum_1to3, matching_1to3, sortedCorrMap_1,
                            mean_X_1, mean_Y_1, mean_X_3, mean_Y_3, 
                            rank_sum_thr = rankSumThr, use_gpu = use_gpu)
    guideflow_3to1, matching_color_3to1 = get_guidance_flow(labelMap3, labelMap1, I3, I1,
                            rankSum_3to1, matching_3to1, sortedCorrMap_3.transpose(1,0), 
                            mean_X_3, mean_Y_3, mean_X_1, mean_Y_1, 
                            rank_sum_thr = rankSumThr, use_gpu = use_gpu)
    
    return guideflow_1to3, guideflow_3to1

input_path = "/home/michael/hilbert/Desktop/Datasets/temp.mp4"
checkpoint = os.path.join("checkpoints", "ATD12K-Style", "49.pth")
output_path = "output.mp4"

# Gets input video's metadata
videoCapture = cv2.VideoCapture(input_path)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
total_frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
videoCapture.release()

videogen = skvideo.io.vreader(input_path)
lastframe = next(videogen)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_path_wo_ext, ext = os.path.splitext(input_path)
print('{}.{}, {} frames in total, {}FPS to {}FPS'.format(video_path_wo_ext, ext, total_frames, fps, 2 * fps))

h, w, _ = lastframe.shape
vid_out = cv2.VideoWriter(output_path, fourcc, fps * 2, (w, h))

# Loads AFI model
model = AnimeInterp().to(device)
weights = torch.load(checkpoint)
ret = model.load_state_dict(weights["state_dict"], strict=False)
print(ret)
model.eval()

# tmp = max((64, int(64 / scale)))
# ph = ((h - 1) // tmp + 1) * tmp
# pw = ((w - 1) // tmp + 1) * tmp
# padding = (0, pw - w, 0, ph - h)
pbar = tqdm(total=total_frames)

def clear_write_buffer(write_buffer):
    while True:
        item = write_buffer.get()
        if item is None:
            break
        vid_out.write(item[:, :, ::-1])

def build_read_buffer(read_buffer, videogen):
    try:
        for frame in videogen:
            read_buffer.put(frame)
    except:
        pass
    read_buffer.put(None)

write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)
_thread.start_new_thread(build_read_buffer, (read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

I1 = torch.from_numpy(np.transpose(lastframe, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
temp = None # Save lastframe when processing static frame

while True:
    if temp is not None:
        frame = temp
        temp = None
    else:
        frame = read_buffer.get()
    if frame is None:
        break
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2,0,1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.

    F12i, F21i = generate_sgm(I0.cpu().numpy(), I1.cpu().numpy())
    F12i.float().to(device)
    F21i.float().to(device)

    outputs = model(I0, I1, F12i, F21i, t=0.5)
    It_warp = outputs[0]
    It = revNormalize(It_warp[0].cpu()).clamp(0.0, 1.0).numpy().transpose(1, 2, 0)

    write_buffer.put(lastframe)
    write_buffer.put(It)

    pbar.update(1)
    lastframe = frame

write_buffer.put(lastframe)

import time
while not write_buffer.empty():
    time.sleep(0.1)
pbar.close()
vid_out.release()