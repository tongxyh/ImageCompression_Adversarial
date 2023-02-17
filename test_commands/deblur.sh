cd ..
# python random_noise.py -degrade deblur -s "/workspace/ct/datasets/GOPRO_Large/test/GOPR0384_11_00/blur/*.png" -t "/workspace/ct/datasets/GOPRO_Large/test/GOPR0384_11_00/sharp/*.png" -q 0 -metric mse | grep AVG > ./logs/deblur_baseline_mse &

for num in {1..8}
do
    # python random_noise.py -degrade deblur \
    # -s "/workspace/ct/datasets/GOPRO_Large/test/GOPR0384_11_00/blur/*.png" \
    # -t "/workspace/ct/datasets/GOPRO_Large/test/GOPR0384_11_00/sharp/*.png" \
    # -ckpt ./ckpts/adv/hyper-${num}-mse-0.0001/best_loss.pth.tar -q $num &

    python random_noise.py -degrade deblur \
    -s "./attack/kodak/blur/*.png" \
    -t "/workspace/ct/datasets/kodak/*.png" -metric mse -q $num \
    -ckpt ./ckpts/adv/hyper-${num}-mse-0.0001/best_loss.pth.tar &
    sleep 10s
done
