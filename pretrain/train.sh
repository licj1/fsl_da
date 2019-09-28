srun -p AD --gres=gpu:8 -n1 python -u main_resnet.py --epochs 300 --batch_size 1024 #2>&1 | tee log.txt &
