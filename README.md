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

## References
[Deep Animation Video Interpolation in the Wild](https://arxiv.org/abs/2104.02495)

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)

[FILM: Frame Interpolation for Large Motion](https://arxiv.org/abs/2202.04901)
