## Environment
### Basic
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
### Packages
```bash
conda install opencv scikit-image
```

## References
[Deep Animation Video Interpolation in the Wild](https://arxiv.org/abs/2104.02495)

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)
