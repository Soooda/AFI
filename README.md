## Environment
### Basic
Run under NVCC *10.0.130*.
```bash
conda create --name AFI python=3.7
conda activate AFI
conda install pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0 -c pytorch
```
### Packages
```bash
conda install scikit-image
```

## References
[Deep Animation Video Interpolation in the Wild](https://arxiv.org/abs/2104.02495)

[Learning to Estimate Hidden Motions with Global Motion Aggregation](https://arxiv.org/abs/2104.02409)
