# Robust Neural Image Compression
Deep neural network-based image compression has been extensively studied. However, the model robustness  which is crucial to practical application is largely overlooked.
We propose to examine the robustness of prevailing learned image compression models by injecting negligible adversarial perturbation into the original source image. Severe distortion in decoded reconstruction reveals the general vulnerability in existing methods regardless of compression  settings (e.g., network architecture, loss function, quality scale). We then explore possible defense strategies against the adversarial attack to improve the model robustness, including geometric self-ensemble based pre-processesing, and adversarial training. Experiments report the effectiveness of various defense strategies. Additional image recompression case study further confirms the substantial improvement of the robustness of compression models in real-life applications. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learned image compression solutions.

# User Guide
## Distorion Attack
```
# Balle2016, Balle2018, Minnen, Cheng2020
python attack_rd.py  -m facotorized/hyper/context/cheng2020 -metric mse/ms-ssim -q [1-8] -s <source_image>

# Note that for NLAIC\HiFiC\InvCompress\Weixin2021 you need to first get their original code and pretrained models
# NLAIC
python attack_nlaic.py -q [0-15] -steps 1001 -s <source_image>

# HiFiC
# Run in directory: /ct/code/compression/models
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python3 -m hific.attack_hific --config hific --ckpt_dir ckpts/hific --tfds_dataset_name coco2014 --out_dir ./out -lr 0.1 --images_glob <source_image>

# InvCompress
python -m compressai.utils.attack_inv checkpoint <source_image> -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01 --cuda -lr 0.001 -steps 1001

# Weixin2021
python attack_fic.py -noise 0.0001 -steps 1001 -s <source_image>
```

## Defense
```
# Adversarial Training
python train.py --adv -lr_train 1e-5 -steps 300 -q 1 -n 0.0001 -m hyper -metric ms-ssim > ./logs/hyper-msim1.log 2>&1 &

# ensemble
python self_ensemble.py --defend --adv
```

## Targeted Attack
```
python attack_rd.py -n 0.02 \
-s /ct/code/mnist_png/testing/2/72.png \
-t /ct/code/mnist_png/testing/0/294.png

# attack with ROI
python attack_rd.py --mask <user_defined_bounding_box> -la_bkg 0.01 -n 0.002 \
-s <source_image> \
-t <target_image>

```

# Contact
Feel free to contact us if there are any questions. (Tong Chen, chentong@nju.edu.cn; Zhan Ma, mazhan@nju.edu.cn)

# Citation
```
@ARTICLE{chen2021robust,
  author={Chen, Tong and Ma, Zhan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3276442}}
```
