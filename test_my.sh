
#srun -p Test --gres=gpu:1 -n1 python -u test.py --load snapshot/domain_1shot_20way_res18_adddata_addloss_fc_0.1/iter_10000_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/domain-net
#srun -p Test --gres=gpu:1 -n1 python -u test.py --load snapshot/mini_1shot_16way_res18_adddata_addloss_fc_0.1/iter_30000_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/mini-imagenet
#srun -p Test --gres=gpu:1 -n1 python -u test.py --load snapshot/tiered_5shot_20way_res18_adddata_addloss_fc_0.1/iter_10000_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/tiered-imagenet
#srun -p Test --gres=gpu:1 -n1 python -u test.py --load snapshot/tiered_1shot_20way_res18_adddata_addloss_fc_0.1/iter_14000_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/tiered-imagenet
#srun -p Test --gres=gpu:1 -n1 python -u test.py --load snapshot/mini_5shot_20way_res18_addloss_fc/iter_20000_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/mini-imagenet
#srun -p AD --gres=gpu:1 -n1 python -u test.py --load snapshot/mini_5shot_30way_res18_addloss_fc_0.1/iter_79500_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/mini-imagenet
srun -p AD --gres=gpu:1 -n1 python -u test.py --load snapshot/domain_5shot_20way_res18_addloss_fc_cosine/iter_20000_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/domain-net --shot 5 #--query 4
#srun -p Test --gres=gpu:1 -n1 python -u test.py --load snapshot/domain_5shot_20way_res18_addloss_fc/iter_90000_model.pth.tar --root /mnt/lustre/dingmingyu/Research/da_zsl/dataset/domain-net --query 4
#-x BJ-IDC1-10-10-30-105