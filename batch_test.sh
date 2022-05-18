for i in {1..6};
do
echo "python ${1} -q $i -metric ${2} -m ${4} -ckpt ./ckpts/adv/${4}-$i-${2}-${3}/best_loss.pth.tar"
python ${1} -q $i -metric ${2} -m ${4} -ckpt "./ckpts/adv/${4}-$i-${2}-${3}/best_loss.pth.tar"
done