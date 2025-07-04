import os
import torch
import torch.nn as nn
import math
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
import time
import copy
import logging
from tqdm import tqdm
import random
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
dir = os.getcwd()
ROOT = os.path.dirname(dir)
sys.path.append(ROOT)
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import cifar_10
from datasets.cifar_10 import membership_dataset_loader
from models.lenet import lenet5, lenet5_half

# non-member 
member_idx = np.load('./datasets/cifar-10/member_idx.npy')
nonmember_idx = np.load('./datasets/cifar-10/nonmember_idx.npy')
shadow_idx = np.load('./datasets/cifar-10/shadow_idx.npy')
member, nonmember, shadow = membership_dataset_loader(member_idx, nonmember_idx, shadow_idx)

# test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = datasets.CIFAR10(root='/cache/data/', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=1024, shuffle=False)

def accuracy(output, target, topk=(1,)):
    '''
    topk = 1이라면 가장 높은 예측 확률을 가진 레이블과 실제 레이블이 동일한지 계산 
    topk = (1, 5)라면, 가장 높은 예측 확률을 가진 레이블과 실제 레이블이 동일한 경우를 계산하여
    top1 정확도 구하고, 그 다음으로 높은 5개의 예측 확률을 가진 레이블 중 실제 레이블이 포함되는지 확인하여 top5 정확도 구함
    
    더욱 모델의 성능을 상세하게 평가하기 위한 방법으로, 모델의 성능을 다각도로 이해하고 평가하는 데 도움됨
    '''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(model, criterion, train_loader, optimizer):

    model.train()
    total_loss = 0
    total_acc1 = 0
    total_acc5 = 0
    total_samples = 0
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training", leave=False) as pbar:
        for idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_size = inputs.size(0)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()    
            # loss.item()은 loss값을 스칼라로 반환
            # _, predicted = outputs.max(1) outputs.max(1)은 각 입력 샘플에 대해 가장 큰 값과 해당 인덱스 반환
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            total_loss += loss.item() * batch_size
            total_acc1 += acc1.item() * batch_size
            total_acc5 += acc5.item() * batch_size
            total_samples += batch_size        
    train_loss = total_loss / total_samples
    train_acc1 = total_acc1 / total_samples
    train_acc5 = total_acc5 / total_samples
        
    return train_loss, train_acc1, train_acc5

def test(model, criterion, test_loader):
    model.eval()
    test_loss = 0
    test_acc1 = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            batch_size = inputs.size(0)
            
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            test_acc1 += acc1.item() * batch_size
            test_loss += loss.item() * batch_size
            total_samples += batch_size

        test_acc1 = test_acc1 / total_samples
        test_loss = test_loss / total_samples
        
        return test_loss, test_acc1
    
def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def model_train(model, model_name, dataset):
    print(f"device:{device}")
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    start_time = time.time()
    logging.basicConfig(filename=f"./student_train_log/cifar-10_{now}_{model_name}.log", level=logging.INFO, format='%(asctime)s - %(message)s')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    model.to(device)
    
    best_acc = 0
    epoch_length = 200
    
    for epoch in range(epoch_length):
        adjust_learning_rate(optimizer, epoch)
        logging.info(f"Epoch: {epoch + 1} / {epoch_length}")
        print(f"epoch : {epoch + 1} / {epoch_length}")
        train_loss, train_acc1, train_acc5 = train(model, criterion, dataset, optimizer)
    
        if train_acc1 > best_acc:
            best_acc = train_acc1
            best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        
        print(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
        logging.info(f"Train Loss: {train_loss}, Top-1 Accuracy: {train_acc1}, Top-5 Accuracy: {train_acc5}")
        
        test_loss, test_acc1 = test(model, criterion, test_loader)
        print(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}")
        logging.info(f"Test Loss: {test_loss}, Top-1 Accuracy: {test_acc1}")
        
    
    learning_time = time.time() - start_time
    logging.info(f"Learning Time: {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    
    torch.save(best_model_wts, f"model_pth/best_model_weights_{model_name}_student_cifar_10.pth")
    print(f"Learning Time : {learning_time // 60:.0f}m {learning_time % 60:.0f}s")
    
if __name__ == "__main__":
    model, model_name = lenet5_half()
    model_train(model, model_name, nonmember)