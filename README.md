# Deep Animation Video Interpolation with Global Motion Aggregation and Style Loss

This is the code repository for my bachelor's graduation thesis. You can view my thesis via [this](https://hilbertkong.me/assets/files/Thesis.pdf).

## Environment

> This project was once trained on The University of Sydney's HPC Artemis which had a very ancient version of Pytorch and other frameworks. If you are under similar circumstances, please follow these procedures:

<details>
	<summary>Click to Reveal</summary>

Run under NVCC *10.0.130*.
```bash
$ conda create -n AFI pip python=3.7
$ conda activate AFI
$ conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
$ conda install scipy opencv
$ pip install einops easydict

$ pip install torch-scatter -f https://data.pyg.org/whl/torch-1.4.0+cu100.html
```

`skimage.measure.compare_psnr()` and `skimage.measure.compare_ssim()` was replaced by `skimage.metrics.peak_signal_noise_ratio()` and `skimage.metrics.structural_similarity()`.

Ref. [scikit-image Official Doc 0.16.1](https://scikit-image.org/docs/0.16.x/api/skimage.measure.html?highlight=compare_psnr#skimage.measure.compare_psnr)

Install lower version to avoid modifications.
```bash
$ conda install scikit-image==0.14.3 -c conda-forge
```

> If you got:
```text
ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'
```
Try:
```bash
$ conda install pillow=6.1
```
</details>

---

Otherwise, just follow these:
```bash
$ conda create -n AFI pip python=3.10
$ conda install pytorch==1.12.1 cudatoolkit=11.3 torchvision==0.13.1 -c pytorch
$ conda install scipy scikit-image
$ conda install pytorch-scatter -c pyg
$ pip install einops easydict
```

A very recent bug is that if you use `conda` to install OpenCV, it will automatically degrade your PyTorch to the CPU version. So please use `pip` instead for installation.
```bash
$ pip install opencv-python
```

`skimage.measure.compare_psnr()` and `skimage.measure.compare_ssim()` was replaced by `skimage.metrics.peak_signal_noise_ratio()` and `skimage.metrics.structural_similarity()`.

Ref. [scikit-image Official Doc 0.16.1](https://scikit-image.org/docs/0.16.x/api/skimage.measure.html?highlight=compare_psnr#skimage.measure.compare_psnr)

Install lower version to avoid modifications.
```bash
$ conda install scikit-image==0.14.3 -c conda-forge
```

> If you got:
```text
ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'
```
Try:
```bash
$ conda install pillow=6.1
```

## Data

Click the links below to download the corresponding datasets:

* [QVI960](https://www.dropbox.com/s/4i6ff6o62sp2f69/QVI-960.zip?dl=0)
* [ATD-12K](https://entuedu-my.sharepoint.com/:u:/g/personal/siyao002_e_ntu_edu_sg/EY3SG0-IajxKj9HMPz__zOMBvyJdrA-SlwpyHYFkDsQtng?e=q7nGlu)
* [ATD-12K Training Set SGM]() (The link is temporarily unavailable.)

> You can also generate the SGM flows yourself. 
> The content of the training SGM flows are extacted to `datasets/atd-12k`.

So the final hierarchy will be like this:

```
datasets
  +--- QVI-960
  +--- atd-12k
          +--- train_10k_pre_calc_sgm_flows
          +--- test_2k_540p
          +--- ...
```

## Weights

The qualitative and quantitative results in the paper were based on a not fully trained model. I have trained the model fully now on a single RTX3090. All the weights are provided in the `checkpoints` folder.

`animeinterp+gma.pth` is the intialised weights of the model combined with pretained weights from [AnimeInterp](https://github.com/lisiyao21/AnimeInterp) and [GMA](https://github.com/zacjiang/GMA). This is the starting point of the training.

`QVI960/199.pth`: The synthesis module of the model is then fine-tuned in QVI-960 datasets for 200 epochs.

`ATD12K/49.pth`: Starting with `QVI960/199.pth`, the whole model is then trained on L1 loss for 50 epochs.

`Style/49.pth`: Starting with `QVI960/199.pth`, the whole model is then trained directly on the Style loss for 50 epochs.

`ATD12K-Style`: Starting with `ATD12K/49.pth`, the whole model is then trained on the Style loss for another 50 epochs.

## Train

Run the command below to start training on QVI-960 dataset:

```bash
$ python QVI960_train.py
```

Run the command below to start training on ATD-12K dataset using L1 Loss:

```bash
$ python ATD12K_train.py
```

Run the command below to start training on ATD-12K dataset using Style Loss:

```bash
$ python ATD12K_train_style.py
```

## Evaluation

Modify `configs/config_test_w_sgm.py` to specify the configuration fo the evaluation process, then run:

```bash
$ python test_anime_sequence_one_by_one.py configs/config_test_w_sgm.py
```

## References

[Deep Animation Video Interpolation in the Wild](https://arxiv.org/abs/2104.02495)

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)

[FILM: Frame Interpolation for Large Motion](https://arxiv.org/abs/2202.04901)
