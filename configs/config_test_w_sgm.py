
testset_root = 'datasets/atd-12k/test_2k_540p'
test_flow_root = 'datasets/atd-12k/test_2k_pre_calc_sgm_flows'
test_annotation_root = 'datasets/atd-12k/test_2k_annotations'

test_size = (960, 540)
test_crop_size = (960, 540)

mean = [0., 0., 0.]
std  = [1, 1, 1]

inter_frames = 1

model = 'AnimeInterp'
pwc_path = None

checkpoint = 'checkpoints/ATD12K-Style/49.pth'

store_path = 'outputs/atd-12k'
