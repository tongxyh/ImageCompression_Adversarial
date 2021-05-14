# Attack Learned Image Compression
```
# Distorion Attack
python attack_rd.py -m hyper -metric ms-ssim -q 1 -la 0.2 -step 10001 \
-s /workspace/ct/datasets/kodak/kodim01.png

# Targeted Attack
python attack_rd.py -m nonlocal -cn 1000000 -l 16 -j MSSSIM_rctx --ctx -la 0.75 -step 10001 \
-s /ct/code/mnist_png/testing/9/281.png \
-t /ct/code/mnist_png/testing/0/294.png
```

