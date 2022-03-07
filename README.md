# Multimodal-FFM-TLD
This repository provides a PyTorch implementation of ["Attention-based Multimodal Image Feature Fusion Module for Transmission Line Detection"](https://ieeexplore.ieee.org/abstract/document/9699431), which is accepted by IEEE Transactions on Industrial Informatics.

If you use this code, please cite the paper.
```
@ARTICLE{9699431,
  author={Choi, Hyeyeon and Yun, Jong Pil and Kim, Bum Jun and Jang, Hyeonah and Kim, Sang Woo},
  journal={IEEE Transactions on Industrial Informatics}, 
  title={Attention-based Multimodal Image Feature Fusion Module for Transmission Line Detection}, 
  year={2022},
  pages={1-1},
  doi={10.1109/TII.2022.3147833}}
```

## Data set 
We constructed the `Visible Light and Infrared Transmission Line Datset (VITLD)`. The dataset is available at [https://bit.ly/3FBYjBY](https://bit.ly/3FBYjBY).


## Pre-trained Models
UNet with Early Fusion (EF) method of our paper: 
- [Unet_original_4c](https://drive.google.com/file/d/1ni79hc3q24M9OSzc9FnXZ9WglqAS9QtZ/view?usp=sharing)

UNet with proposed feature fusion module (FFM): 
- [Unet_proposed](https://drive.google.com/file/d/1OnYrFX8vKbppSWvb6QpGGcLWikgjjly5/view?usp=sharing)
