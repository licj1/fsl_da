now=`date +%Y-%m-%d,%H:%m:%s`
srun -p AD  -x BJ-IDC1-10-10-30-105 --gres=gpu:8 -n1 --job-name=cdan python -u train_image.py --gpu_id 0,1,2,3,4,5,6,7 --net ResNet50 --dset imagenet --t_dset_path /mnt/lustre/dingmingyu/Research/da_zsl/dataset/mini-imagenet/test_transfer_15.txt --s_dset_path /mnt/lustre/dingmingyu/Research/da_zsl/dataset/mini-imagenet/trainval_list.txt --output_dir output_15 2>&1|tee logs/mini-train15-${now}.log 