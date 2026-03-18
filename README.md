# AOT_GAN_custom
This repository contains code for training and image reconstruction with an AOT_GAN model.  


## Environment setting
Please set up the environment by following the instructions in the official AOT-GAN repository: [https://github.com/facebookresearch/mae.git](https://github.com/researchmm/AOT-GAN-for-Inpainting.git)


## Data Preparation
Organize the image data in the following structure:
```
| data
    | -- train
        | -- 00001_image_000001.png
        | -- 00001_image_000002.png
        ...
    | -- val
        | -- 00004_image_000001.png
        | -- 00004_image_000002.png
        ...
```


## Training
The script for training the image reconstruction task is `src/train.py`.  
AOT-GAN is trained from scratch on the SLP dataset.

Then run training as follows.
The hyperparameters are specified in the command below, and training logs as well as checkpoint files (tfevents) for each epoch will be saved during training.
```
cd src
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
```


## Inference
Use `src/aot_recon.py` to perform image reconstruction with a trained model.

In addition, the following outputs are generated: the input image, masked image, reconstructed image, and visible image, where the visible patches in the masked image are combined with the reconstructed image.  
The checkpoint path used for image reconstruction, the path to the original images, the path to the mask images, and the path for saving the output images can all be edited directly in the code.

```
cd src
python aot_recon.py
```


## Evaluation
Use `metric.py` in `mae_custom` to measure PSNR, SSIM, and MSE for the generated images.  
Use `_orig.png` as the ground-truth image to evaluate the metrics of `_masked.png` and `_visible.png`.


## Acknowledgement
This repository is based on the official AOT_GAN repository: [https://github.com/facebookresearch/mae.git](https://github.com/researchmm/AOT-GAN-for-Inpainting.git)
