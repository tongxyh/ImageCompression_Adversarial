# Robust Learned Image Compression
model_dir 的默认值为 $TORCH_HOME/models 其中 $TORCH_HOME 默认值为 ~/.torch. 可以使用 $TORCH_MODEL_ZOO 环境变量来覆盖默认目录
export TORCH_HOME=/workspace/ct/code/LearnedCompression/ckpts/balle

## Distorion Attack
```

# Balle
python attack_rd.py -m hyper -metric ms-ssim -q 1 -la 0.2 -step 10001 \
-s /workspace/ct/datasets/kodak/kodim01.png --pretrained

# NLAIC
python attack_rd.py -m nonlocal -cn 1000000 -l 16 -j MSSSIM_rctx --ctx -la 0.2 -step 10001 \
-s /ct/code/mnist_png/testing/9/281.png --d

python attack_nlaic.py  -la 0.5 -step 10001 -s /ct/code/mnist_png/testing/9/281.png
```

## Targeted Attack
```
## random select in 0
# target = "/ct/code/mnist_png/testing/0/294.png"
## random select in [1-9]
# source = "/ct/code/mnist_png/testing/9/281.png" lambda = 1.0
# source = "/ct/code/mnist_png/testing/8/110.png" lambda = 1.0
# source = "/ct/code/mnist_png/testing/7/411.png" lambda = 0.8
# source = "/ct/code/mnist_png/testing/6/940.png" lambda = 1.0
# source = "/ct/code/mnist_png/testing/5/509.png" lambda = 1.0
# source = "/ct/code/mnist_png/testing/4/109.png" lambda = 1.0
# source = "/ct/code/mnist_png/testing/3/1426.png"lambda = 1.0
# source = "/ct/code/mnist_png/testing/2/72.png"  lambda = 1.0
# source = "/ct/code/mnist_png/testing/1/430.png" lambda = 1.0

# MNIST
python attack_rd.py -m factorized -q 1 -metric mse -la 1.0 -step 10001 --d \
-s /ct/code/mnist_png/testing/9/281.png \
-t /ct/code/mnist_png/testing/0/294.png

# Cityscapes
python attack_rd.py -m factorized -q 1 -metric mse -la 1.0 -step 10001 --d --mask 112 199 103 137 \
-s ./attack/licenseplate/MZ2837_origin.png \
-t ./attack/licenseplate/MZ8723_origin.png

python attack_rd.py -m hyper -la_bkg 0.0 -q 3 -metric ms-ssim -la 0.2 -step 10001 --d --mask 112 199 103 137 -s ./attack/licenseplate/MZ2837_origin.png -t ./attack/licenseplate/MZ8723_origin.png -lr 1e-4
python attack_rd.py -m hyper -la_bkg 0.0 -q 3 -metric ms-ssim -la 0.2 -step 10001 --d --mask 56 100 51 69 -s ./attack/licenseplate/MZ2837_120x120.png -t ./attack/licenseplate/MZ8723_120x120.png -lr 1e-4

python attack_rd.py -m factorized -la_bkg 0.25 -q 4 -metric ms-ssim -la 0.175 -step 10001 --d --mask 112 199 103 137 -s ./attack/licenseplate/MZ2837_origin.png -t ./attack/licenseplate/MZ2222_origin.png -lr 1e-3
python attack_rd.py -m hyper -la_bkg 0.25 -q 4 -metric ms-ssim -la 0.22 -step 10001 --d --mask 56 100 51 69 -s ./attack/licenseplate/MZ2837_120x120.png -t ./attack/licenseplate/MZ2222_120x120.png -lr 1e-3
```


## Data Augmentation
```
python attack_data.py -la 0.2 -step 1001 -m hyper -q 3

python train.py -m hyper -la 0.1 -q 3 -lr 1e-5 -ckpt ./ckpts/attack/anchor/xxx --pretrained
python train.py -m hyper -la 0.1 -q 3 -metric mse -lr 1e-4 -ckpt ./ckpts/attack/anchor/xxx

python visual.py -m hyper -metric ms-ssim -q 2 -s /workspace/ct/datasets/kodak/kodim10.png --d
python visual.py -m hyper -metric ms-ssim -q 3 -s ./attack/kodak/fake3333_ -ckpt ./ckpts/attack/anchor/hyper-3-fromscratch//ae_100_0_0.02102314_0.14399883.pkl
```

## test
```
python visual.py -m hyper -metric ms-ssim -q 2 -s /workspace/ct/datasets/kodak/kodim10.png --d

#HiFiC
#https://storage.googleapis.com/tensorflow_compression/metagraphs/models.txt
python3 -m hific.evaluate_attacker --config hific --ckpt_dir ckpts/hific --out_dir out/ \
                   --tfds_dataset_name coco2014 --images_filename 
```

## Visual Distribution
```
/workspace/ct/datasets/datasets/div2k/
python visual_distribution.py -m hyper -metric ms-ssim -q 2 -s "/workspace/ct/datasets/attack/hyper-2/adversarial/*.png" --d
python visual_distribution.py -m hyper -metric ms-ssim -q 2 -s "/workspace/ct/datasets/datasets/div2k/*.png" --d
```
## JPEG sr6 anchor
```
# DONT SUPPORT PNG format
cjpeg -q 50 -outfile ./kodim08.jpg /workspace/ct/datasets/kodak/bmp/kodim08.bmp
djpeg -v -bmp -outfile kodim08.bmp ./kodim08.jpg
```