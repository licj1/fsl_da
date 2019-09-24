srun -p AD --gres=gpu:7 -n1 python -u main_resnet.py --epochs 300 --batch_size 128 #2>&1 | tee log.txt &
