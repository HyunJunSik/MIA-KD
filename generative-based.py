#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.


'''
python generative-based.py --dataset cifar10 --channels 3 --n_epochs 500 --batch_size 1024 --lr_G 0.02 --lr_S 0.1 --latent_dim 1000 --oh 0.05 --ie 5 --a 0.01
'''
import argparse
import os
import numpy as np
import math
import sys
import pdb
import copy

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
import torchvision
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
dir = os.getcwd()
ROOT = os.path.dirname(dir)
sys.path.append(ROOT)
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
from models.resnet import resnet20, resnet56
from datasets import cifar_10

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--teacher_dir', type=str, default='./model_pth/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.02, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=150, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default='./generator_pth/')

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True 

def show_images(images, title=""):
    images = images.detach().cpu()
    grid_img = torchvision.utils.make_grid(images, nrow=8, normalize=True)
    plt.figure(figsize=(8, 2))
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=False),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            # nn.BatchNorm2d(opt.channels, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img
        
generator = Generator().cuda()
# torch.save(net,args.output_dir + '_' + net_name + '_' + args.dataset)
# teacher = torch.load(opt.teacher_dir + 'resnet34_' + opt.dataset).cuda()
teacher, teacher_name = resnet56(num_classes=10)
teacher.load_state_dict(torch.load(opt.teacher_dir + 'best_model_weights_resnet56_teacher_cifar_10.pth'))
teacher.cuda()
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

teacher = nn.DataParallel(teacher)
generator = nn.DataParallel(generator)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl


if opt.dataset == 'MNIST':    
    # Configure data loader   
    net, net_name = resnet20(num_classes=10)
    net.cuda()
    net = nn.DataParallel(net)
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))           
    data_test_loader = DataLoader(data_test, batch_size=64, num_workers=0, shuffle=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
    optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)


# test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if opt.dataset != 'MNIST':  
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if opt.dataset == 'cifar10': 
        net, net_name = resnet20(num_classes=10)
        net.cuda()
        net = nn.DataParallel(net)
        data_test = datasets.CIFAR10(root='/cache/data/', train=False, download=True, transform=transform)
    if opt.dataset == 'cifar100': 
        net, net_name = resnet20(num_classes=100)
        net.cuda()
        net = nn.DataParallel(net)
        data_test = CIFAR100(opt.data,
                          train=False,
                          transform=transform_test)
    data_test_loader = DataLoader(data_test, batch_size=opt.batch_size, num_workers=0)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)

    optimizer_S = torch.optim.SGD(net.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)


best_acc = 0
best_model_wts = copy.deepcopy(net.state_dict())

def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):

    total_correct = 0
    avg_loss = 0.0
    if opt.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, opt.lr_S)

    for i in range(120):
        net.train()
        z = Variable(torch.randn(opt.batch_size, opt.latent_dim)).cuda()
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()        
        gen_imgs = generator(z)
        outputs_T, features_T = teacher(gen_imgs)   
        features_T = features_T["pooled_feat"][-1]
        pred = outputs_T.data.max(1)[1]
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss = loss_one_hot * opt.oh + loss_information_entropy * opt.ie + loss_activation * opt.a
        
        gen_logit, gen_feat = net(gen_imgs.detach())
        
        loss_kd = kdloss(gen_logit, outputs_T.detach()) 
        loss += loss_kd       
        loss.backward()
        optimizer_G.step()
        optimizer_S.step() 
        if i == 1:
            print ("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))
            
    net.eval()
    with torch.no_grad():    
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            output, _ = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test_loader)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    
    if accr > best_acc:
        best_acc = accr
        best_model_wts = copy.deepcopy(net.state_dict())
        torch.save(best_model_wts, f"model_pth/best_model_weights_{net_name}_student_{opt.dataset}.pth")

    if (epoch + 1) % 50 == 0:
        checkpoint_path = f"./checkpoints/{net_name}_generator.pth"
        torch.save({
            'epoch' : epoch + 1,
            'generator_state_dict' : generator.state_dict(),
            'optimizer_G_state_dict' : optimizer_G.state_dict(),
            'optimizer_S_state_dict' : optimizer_S.state_dict(),
            'best_acc' : best_acc
        }, checkpoint_path)
        print(f"Checkpoint saved")

net.load_state_dict(best_model_wts)

z = Variable(torch.randn(128, opt.latent_dim)).cuda()
gen_imgs = generator(z)
show_images(gen_imgs, title="generated images")
torch.save(generator, opt.output_dir + 'adv_generator_latdim_150' + '_MNIST')

# checkpoint_epoch = 50  # 예: 50 epoch에서 재개
# checkpoint_path = f"imagenet_model_pth/checkpoint_epoch_{checkpoint_epoch}.pth"

# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# start_epoch = checkpoint['epoch']
# best_acc = checkpoint['best_acc']

# print(f"Resuming training from epoch {start_epoch}")