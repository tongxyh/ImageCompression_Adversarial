## -*- coding: utf-8 -*-
import sys
import metrics
from glob import glob

a, b = {}, {}
imgs_a = sorted(glob(sys.argv[1]))
imgs_b = sorted(glob(sys.argv[2]))

for i, img in enumerate(imgs_a):
    a[i] = img

for i, img in enumerate(imgs_b):
    b[i] = img    
# a = {0:sys.argv[1]}
# b = {0:sys.argv[2]}
# print(a , b)
# for im0, im1 in zip(sorted(glob(a)), sorted(glob(b))):
# print(im0, im1)
results = metrics.evaluate(a, b)
print(results)