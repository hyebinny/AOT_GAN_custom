# AOT-GAN modified for SLP dataset
## Train
Train script and hyper-parameters:
`
python train.py \
    --save_dir /mnt/d/hyebin/aot-gan-for-inpainting/experiments/slp \
    --dir_image /mnt/d/hyebin/aot-gan-for-inpainting/data/slp_256 \
    --data_train slp_train \
    --data_test slp_test \
    --mask_type random_rec \
    --image_size 256 \
    --batch_size 10 \
    --save_every 10000 \
    --print_every 10000 \
    --iterations 500000 \
    --resume \
    --tensorboard
`

<!-- ------------------------------------------------ -->
## Acknowledgement
This repository is modified from: 
![aotgan](https://github.com/researchmm/AOT-GAN-for-Inpainting/blob/master/docs/aotgan.PNG?raw=true)
### [Arxiv Paper](https://arxiv.org/abs/2104.01431) |
AOT-GAN: Aggregated Contextual Transformations for High-Resolution Image Inpainting<br>
[Yanhong Zeng](https://sites.google.com/view/1900zyh),  [Jianlong Fu](https://jianlong-fu.github.io/), [Hongyang Chao](https://scholar.google.com/citations?user=qnbpG6gAAAAJ&hl),  and [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).<br>

```
@inproceedings{yan2021agg,
  author = {Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang and Guo, Baining},
  title = {Aggregated Contextual Transformations for High-Resolution Image Inpainting},
  booktitle = {Arxiv},
  pages={-},
  year = {2020}
}
```

