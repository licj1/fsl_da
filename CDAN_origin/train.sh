now=`date +%Y-%m-%d,%H:%m:%s`
srun -p AD -x BJ-IDC1-10-10-30-101,BJ-IDC1-10-10-30-222 --gres=gpu:8 -n1 --job-name=cdan python -u train_image.py --gpu_id 0,1,2,3,4,5,6,7 --net ResNet50 --dset imagenet 2>&1|tee logs/train-${now}.log 
