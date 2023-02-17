q=50

for num in {1..24}
do
    cjpeg -q ${q} -outfile ./attack/kodak/jpeg_q${q}/kodim0${num}.jpg /workspace/ct/datasets/kodak/bmp/kodim0${num}.bmp
done
