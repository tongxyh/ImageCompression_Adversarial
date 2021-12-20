# Towards Robust Neural Image Compression: Adversarial Attack and Finetuning

## (Untargeted) Distorion Attack
```
# Balle2016, Balle2018, Minnen, Cheng2020
python attack_rd.py -m hyper -metric ms-ssim -q 3 -steps 10001 -s xxx.png --download -noise 0.001

# NLAIC
python attack_nlaic.py -q 8 -steps 10001 -s xxx.png

# HiFiC
# yun.nju.edu.cn:5000/chentong/tensorflow:1.15.2-cuda10.0-cudnn7-py36-runtime
# run in '/ct/code/compression/models' on 2080-1
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true python3 -m hific.attack_hific --config hific --ckpt_dir ckpts/hific --tfds_dataset_name coco2014 --out_dir ./out -lr 0.1 --images_glob /ct/datasets/kodak/kodim01.png

# InvCompress
# Note: replace original InvCompress/codes/compressai directory with compiled files in /workspace/InvCompress/codes/compressai
python -m compressai.utils.attack_inv checkpoint /workspace/ct/datasets/kodak/kodim01.png -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01 --cuda -lr 0.001 -steps 10001

# Weixin
# \ct\code\fixed-point-main\quant_4
python attack_fic.py -noise 0.001 -steps 10001 -s xxx.png 
```

## Targeted Attack
```
python attack_rd.py -m factorized -q 1 -metric mse -steps 10001 -n 0.02 --download \
-s /ct/code/mnist_png/testing/2/72.png \
-t /ct/code/mnist_png/testing/0/294.png

# Cityscapes
python attack_rd.py -m hyper -q 3 -metric ms-ssim -steps 10001 --download --mask 112 199 103 137 -la_bkg 0.01 -n 0.002 \
-s ./attack/licenseplate/MZ2837_origin.png \
-t ./attack/licenseplate/MZ2222_origin.png

python attack_rd.py -m hyper -la_bkg 0.0 -q 3 -metric ms-ssim -la 0.2 -steps 10001 --download --mask 112 199 103 137 -s ./attack/licenseplate/MZ2837_origin.png -t ./attack/licenseplate/MZ8723_origin.png -lr 1e-4
python attack_rd.py -m hyper -la_bkg 0.0 -q 3 -metric ms-ssim -la 0.2 -steps 10001 --download --mask 56 100 51 69 -s ./attack/licenseplate/MZ2837_120x120.png -t ./attack/licenseplate/MZ8723_120x120.png -lr 1e-4

python attack_rd.py -m factorized -la_bkg 0.25 -q 4 -metric ms-ssim -la 0.175 -steps 10001 --download --mask 112 199 103 137 -s ./attack/licenseplate/MZ2837_origin.png -t ./attack/licenseplate/MZ2222_origin.png -lr 1e-3
python attack_rd.py -m hyper -la_bkg 0.25 -q 4 -metric ms-ssim -la 0.22 -steps 10001 --download --mask 56 100 51 69 -s ./attack/licenseplate/MZ2837_120x120.png -t ./attack/licenseplate/MZ2222_120x120.png -lr 1e-3
```


## Iterative Adversarial Finetuning 
```
python adv_train.py -m hyper -q 3 -la 8.73 --download -steps 1000 -n 0.001 -lr 0.01
```

## Contact
Feel free to contact us if there is any question. (Tong Chen, tong@smail.nju.edu.cn; Zhan Ma, mazhan@nju.edu.cn)

## Citation
```
@misc{chen2021robust,
      title={Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning}, 
      author={Tong Chen and Zhan Ma},
      year={2021},
      eprint={2112.08691},
      archivePrefix={arXiv},
}
```