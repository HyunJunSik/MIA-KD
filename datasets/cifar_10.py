import torch
from torch.utils.data import DataLoader, Subset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

full_trainset = datasets.CIFAR10(root='/cache/data/', train=True, download=True, transform=transform)
# full_testset = datasets.CIFAR10(root='/cache/data/', train=False, download=True, transform=transform)

# 전체 trainset 섞기
train_idx = np.arange(len(full_trainset))
np.random.shuffle(train_idx)

# 훈련 데이터셋에서 member / nonmember를 33%씩 분할, shadow는 나머지 
num_train = len(full_trainset)
member_size = num_train // 3
nonmember_size = num_train // 3
shadow_size = num_train - member_size - nonmember_size

member_idx = train_idx[:member_size]
nonmember_idx = train_idx[member_size:member_size+nonmember_size]
shadow_idx = train_idx[member_size+nonmember_size:]

def membership_dataset_loader(mem_idx, nonmem_idx, shadow_idx):
    member_dataset = Subset(full_trainset, mem_idx)
    nonmember_dataset = Subset(full_trainset, nonmem_idx)
    shadow_dataset = Subset(full_trainset, shadow_idx)
    
    mem_loader = DataLoader(member_dataset, batch_size=1024, shuffle=True)
    nonmem_loader = DataLoader(nonmember_dataset, batch_size=1024, shuffle=True)
    shadow_loader = DataLoader(shadow_dataset, batch_size=1024, shuffle=True)
    
    return mem_loader, nonmem_loader, shadow_loader

if __name__=="__main__":
    np.save('member_idx.npy', member_idx)
    np.save('nonmember_idx.npy', nonmember_idx)
    np.save('shadow_idx.npy', shadow_idx)
    
    member, nonmember, shadow = membership_dataset_loader(member_idx, nonmember_idx, shadow_idx)
    print(member)