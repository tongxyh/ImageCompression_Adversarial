# encode with name
import os
from glob import glob

# CLIC Valid
images = glob("/data/chentong/CLIC2020_valid/valid/*.png")
# Kodak
# images = glob("/data/ljp105/NIC_Dataset/test/ClassD_Kodak/*.png")

for image in images:
    filename = image.split('/')[-1]
    cmd = "python inference.py --encode -i %s -o /output/%s -m_dir /model/chentong/NIC_rctx --block_width 3000 --block_height 3000 -m 0"%(image, filename[:-3]+"bin")
    os.system(cmd)
    cmd = "python inference.py --decode -i /output/%s -o /output/%s -m_dir /model/chentong/NIC_rctx --block_width 3000 --block_height 3000 -m 0"%(filename[:-3]+"bin", filename[:-4]+"_rec.png")
    os.system(cmd)

    