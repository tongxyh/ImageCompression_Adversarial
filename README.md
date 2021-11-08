# Robust Neural Image Compression
The default `model_dir` is `$TORCH_HOME/models`, when `$TORCH_HOME` is set to `~/.torch` in default. You may use `$TORCH_MODEL_ZOO` environment variable to overwrite the default directory.
```
export TORCH_HOME=/workspace/ct/code/LearnedCompression/ckpts/balle
```

## Distorion Attack
```
# Balle2016, Balle2018, Minnen, Cheng2020
python attack_rd.py -m hyper -metric ms-ssim -q 1 -la 0.2 -step 10001 -s /workspace/ct/datasets/kodak/kodim01.png --download

# NLAIC
python attack_nlaic.py -q 8 -step 10001 -s /workspace/ct/datasets/kodak/kodim20.png

# python attack_rd.py -m nonlocal -cn 1000000 -l 16 -j MSSSIM_rctx --ctx -la 0.2 -step 10001 \
-s /ct/code/mnist_png/testing/9/281.png --download

# HiFiC
# yun.nju.edu.cn:5000/chentong/tensorflow:1.15.2-cuda10.0-cudnn7-py36-runtime
TF_FORCE_GPU_ALLOW_GROWTH=true python3 -m hific.attack_hific --config mselpips --ckpt_dir ckpts/mse_lpips --tfds_dataset_name coco2014 --out_dir ./out

# InvCompress
# Note: replace original InvCompress/codes/compressai directory with compiled files in /workspace/InvCompress/codes/compressai
python -m compressai.utils.attack_inv checkpoint /workspace/ct/datasets/kodak/kodim01.png -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01 --cuda

# TIC
python -m attack_TIC.py checkpoint /workspace/ct/datasets/kodak/kodim01.png -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01 --cuda
```

## Targeted Attack
```
# MNIST
## target image: "/ct/code/mnist_png/testing/0/294.png"
## source image: in [1-9]
# "/ct/code/mnist_png/testing/9/281.png", lambda = 1.0
# "/ct/code/mnist_png/testing/8/110.png", lambda = 1.0
# "/ct/code/mnist_png/testing/7/411.png", lambda = 0.8
# "/ct/code/mnist_png/testing/6/940.png", lambda = 1.0
# "/ct/code/mnist_png/testing/5/509.png", lambda = 1.0
# "/ct/code/mnist_png/testing/4/109.png", lambda = 1.0
# "/ct/code/mnist_png/testing/3/1426.png",lambda = 1.0
# "/ct/code/mnist_png/testing/2/72.png",  lambda = 1.0
# "/ct/code/mnist_png/testing/1/430.png", lambda = 1.0
python attack_rd.py -m factorized -q 1 -metric mse -la 1.0 -step 10001 --download \
-s /ct/code/mnist_png/testing/9/281.png \
-t /ct/code/mnist_png/testing/0/294.png

# Cityscapes
python attack_rd.py -m factorized -q 1 -metric mse -la 1.0 -steps 10001 --download --mask 112 199 103 137 \
-s ./attack/licenseplate/MZ2837_origin.png \
-t ./attack/licenseplate/MZ8723_origin.png

python attack_rd.py -m hyper -la_bkg 0.0 -q 3 -metric ms-ssim -la 0.2 -steps 10001 --download --mask 112 199 103 137 -s ./attack/licenseplate/MZ2837_origin.png -t ./attack/licenseplate/MZ8723_origin.png -lr 1e-4
python attack_rd.py -m hyper -la_bkg 0.0 -q 3 -metric ms-ssim -la 0.2 -steps 10001 --download --mask 56 100 51 69 -s ./attack/licenseplate/MZ2837_120x120.png -t ./attack/licenseplate/MZ8723_120x120.png -lr 1e-4

python attack_rd.py -m factorized -la_bkg 0.25 -q 4 -metric ms-ssim -la 0.175 -steps 10001 --download --mask 112 199 103 137 -s ./attack/licenseplate/MZ2837_origin.png -t ./attack/licenseplate/MZ2222_origin.png -lr 1e-3
python attack_rd.py -m hyper -la_bkg 0.25 -q 4 -metric ms-ssim -la 0.22 -steps 10001 --download --mask 56 100 51 69 -s ./attack/licenseplate/MZ2837_120x120.png -t ./attack/licenseplate/MZ2222_120x120.png -lr 1e-3

# CelebA
python attack_rd.py -m factorized -q 1 -metric mse -noise 0.01 -steps 10001 --download \
-s ./attack/face/000016.jpg -t ./attack/face/000012.jpg
```

## Train INN
```
# replace ours.py & our_utils.py in compressai and recompile & reinstall compressai by 'pip install -e .'
# lmabda * D + R
python examples/train.py -exp exp_01_mse_q1 -m invcompress -d /workspace/ct/datasets/datasets/ --epochs 600 -lr 1e-4 --batch-size 8 --cuda --gpu_id 0 --lambda 0.0016 --metrics mse --save 
python -m compressai.utils.update_model -exp exp_01_mse_q1 -a invcompress --epoch xxx
python -m compressai.utils.eval_model checkpoint ./attack/ -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01
```

## Data Augmentation
```
python attack_data.py -la 0.2 -step 1001 -m hyper -q 3

python train.py -m hyper -la 0.1 -q 3 -lr 1e-5 --pretrained
python train.py -m hyper -la 0.1 -q 3 -metric mse -lr 1e-4 -ckpt ./ckpts/attack/anchor/xxx

python visual.py -m hyper -metric ms-ssim -q 2 -s /workspace/ct/datasets/kodak/kodim10.png --download
python visual.py -m hyper -metric ms-ssim -q 3 -s ./attack/kodak/fake3333_ -ckpt ./ckpts/attack/anchor/hyper-3-fromscratch//ae_100_0_0.02102314_0.14399883.pkl
```

## Test
```
python visual.py -m hyper -metric ms-ssim -q 2 -s /workspace/ct/datasets/kodak/kodim10.png -t ./attack/kodak/out.png --download

#HiFiC
#https://storage.googleapis.com/tensorflow_compression/metagraphs/models.txt
python3 -m hific.evaluate_attacker --config hific --ckpt_dir ckpts/hific --out_dir out/ \
                   --tfds_dataset_name coco2014 --images_filename 

#InvCompress
python -m compressai.utils.eval_model checkpoint ./attack/fake_in.png -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01
```

## Visual Distribution
```
/workspace/ct/datasets/datasets/div2k/
python visual_distribution.py -m hyper -metric ms-ssim -q 2 -s "/workspace/ct/datasets/attack/hyper-2/adversarial/*.png" --download
python visual_distribution.py -m hyper -metric ms-ssim -q 2 -s "/workspace/ct/datasets/datasets/div2k/*.png" --download
```
## JPEG sr6 anchor
```
# DO NOT SUPPORT PNG format
cjpeg -q 50 -outfile ./kodim08.jpg /workspace/ct/datasets/kodak/bmp/kodim08.bmp
djpeg -v -bmp -outfile kodim08.bmp ./kodim08.jpg
```
## Contact
Feel free to contact us if there is any question. (Tong Chen, tong@smail.nju.edu.cn; Zhan Ma, mazhan@nju.edu.cn)