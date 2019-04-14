import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from data_loader import *
import CDAN_pytorch.network as network
from prototypical_network_pytorch.utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--load', default='snapshot/san_devide10/iter_09500_model.pth.tar')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=30) 
    parser.add_argument('--root', default='/mnt/lustre/dingmingyu/Research/da_zsl/dataset/tiered-imagenet/')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    # dataset = MiniImageNet('test')
    dataset = MiniImageNet(root=args.root, dataset='tiered-imagenet', mode='test_new_domain_fsl')
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    # model = Convnet().cuda()
    model = torch.load(args.load)
    # model= list(model.children())[0]
    # model = model.module
#     for key, value in base_network.state_dict().items():
#         print(key)
    # model = list(model.children())[9].cuda()
    # base_network= torch.nn.Sequential(*list(base_network.children())[:-1]).cuda()
    model = nn.DataParallel(model)
    print(model)
    model.eval()

    ave_acc = Averager()
    var = []
    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x, _ = model(data_shot)
        x = x.reshape(args.shot, args.way, -1).mean(dim=0)
        p = x

        proto_query, _ = model(data_query)
        logits = euclidean_metric(proto_query, p)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        var.append(acc)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f}), var: {:.3f}'.format(i, ave_acc.item() * 100, acc * 100, np.array(var).var()))
        
        x = None; p = None; logits = None

