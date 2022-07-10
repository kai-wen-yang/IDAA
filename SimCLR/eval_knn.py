import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import os
#from experiment import ex
from model import load_model, save_model
import wandb
from modules import LogisticRegression
import numpy as np
from tqdm import tqdm


def kNN(epoch, net, trainloader, testloader, K, sigma, ndata, low_dim = 128):
    net.eval()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

    try:
        trainLabels = torch.LongTensor(trainloader.dataset.targets).cuda()
    except:
        trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    trainFeatures = np.zeros((low_dim, ndata))
    trainFeatures = torch.Tensor(trainFeatures).cuda() 
    C = trainLabels.max() + 1
    C = np.int(C)
    
    with torch.no_grad(): 
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=256, shuffle=False, num_workers=4)
        for batch_idx, (inputs, targets) in tqdm(enumerate(temploader)):
            targets = targets.cuda()
            batchSize = inputs.size(0)
            _, features = net(inputs.cuda())
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.t()
            
            
    trainloader.dataset.transform = transform_bak
    # 
    
       
    top1 = 0.
    top5 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            targets = targets.cuda()
            batchSize = inputs.size(0)  
            _, features = net(inputs.cuda())
            total += targets.size(0)

            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)

            _, predictions = probs.sort(1, True)
            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1,1))

            top1 = top1 + correct.narrow(1,0,1).sum().item()
            
    print(top1*100./total)

    return top1*100./total