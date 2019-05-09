now=`date +%Y-%m-%d,%H:%m:%s`
srun -p AD   -x BJ-IDC1-10-10-30-105 --gres=gpu:8 -n1 --job-name=cdan python -u eval.py --net ResNet50 --dset imagenet --t_dset_path /mnt/lustre/dingmingyu/Research/da_zsl/dataset/tiered-imagenet/test_transfer_50.txt --output_dir test  2>&1|tee logs/eval-${now}.log 
