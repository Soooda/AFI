# testset_root = '/home/soda/Dataset/atd-12k/test_2k_original'
testset_root = '/home/soda/Dataset/STD_12k/test_2k'
# testset_root = '/home/soda/Dataset/Hakubo'
test_flow_root = '/home/soda/Dataset/atd-12k/test_2k_pre_calc_sgm_flows'
# test_flow_root = '/home/soda/Dataset/Hakubo_pre_calc_sgm_flows'
test_annotation_root = '/home/soda/Dataset/atd-12k/test_2k_annotations'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std  = [1, 1, 1]

inter_frames = 1

model = 'AnimeInterp'
pwc_path = None

checkpoint = 'checkpoints/FAULT-L1/199.pth'

store_path = 'outputs/std-12k-L1'
