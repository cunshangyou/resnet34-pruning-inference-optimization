import torch
from torchvision import datasets, transforms as T
import os

NORMALIZE_DICT = {
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
}


def get_dataset(name: str, data_root: str='datasets', batch_size: int=32):
    data_root = os.path.expanduser( data_root )
    if name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root=data_root+'cifar10'
        train_dst = datasets.CIFAR10(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=False, transform=val_transform)
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root=data_root+'cifar100'
        train_dst = datasets.CIFAR100(data_root, train=True, download=False, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=False, transform=val_transform)
        
    train_loader = torch.utils.data.DataLoader(train_dst, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dst, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    return num_classes,train_loader,val_loader

def get_model_accuarcy(model, val_loader, device):
    correct = 0  
    total = 0  
    with torch.no_grad():  # 不需要计算梯度  
        for images, labels in val_loader:  
            images, labels = images.to(device), labels.to(device)  
            outputs = model(images)  
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  
    return 100 * correct / total
