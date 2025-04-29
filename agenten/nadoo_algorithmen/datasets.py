import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms


def synthetic_gaussian_loader(n_samples=10000, D=128, n_classes=2, batch_size=64, shuffle=True):
    X = torch.randn(n_samples, D)
    y = torch.randint(0, n_classes, (n_samples,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_loaders(name, batch_size=64, data_dir='./data'):
    """
    Returns (train_loader, test_loader) for given dataset name.
    Supported: mnist, fashion_mnist, cifar10, cifar100, svhn, tiny_imagenet
    """
    name = name.lower()
    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif name == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    elif name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    elif name == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
    elif name == 'svhn':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train = datasets.SVHN(data_dir, split='train', download=True, transform=transform)
        test = datasets.SVHN(data_dir, split='test', download=True, transform=transform)
    elif name == 'tiny_imagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        train_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'train')
        val_dir = os.path.join(data_dir, 'tiny-imagenet-200', 'val')
        train = datasets.ImageFolder(train_dir, transform=transform)
        test = datasets.ImageFolder(val_dir, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
