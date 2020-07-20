import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import grad, Variable
from utils.utils import Identity


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def get_dataset(name='tinyimagenet', tensor_type_transformer=Identity):
    """
        return: train_dataset and test_dataset.
    """
    if name == 'mnist':
        num_classes = 10
        train_dataset = datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                tensor_type_transformer(),
            ]))
        test_dataset = datasets.MNIST(
            '../data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                tensor_type_transformer(),
            ]))
    elif name == 'svhn':
        num_classes = 10
        train_dataset = datasets.SVHN(
            '../data',
            split='extra',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.SVHN(
            '../data',
            split='test',
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]))

    elif name == 'cifar10':
        num_classes = 10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(
            root='../data/cifar-10-batches-py',
            train=True,
            download=True,
            transform=transform_train)

        test_dataset = datasets.CIFAR10(
            root='../data/cifar-10-batches-py',
            train=False,
            download=False,
            transform=transform_test)

    elif name == 'cifar100':

        num_classes = 100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            tensor_type_transformer(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR100(
            root='../data/cifar-100-python/train',
            train=True,
            download=True,
            transform=transform_train)

        test_dataset = datasets.CIFAR100(
            root='../data/cifar-100-python/test',
            train=False,
            download=False,
            transform=transform_test)

    elif name == 'tinyimagenet':
        num_classes = 200
        normalize = transforms.Normalize(
            mean=[
                0.44785526394844055, 0.41693055629730225, 0.36942949891090393
            ],
            std=[0.2928885519504547, 0.28230994939804077, 0.2889912724494934])
        
        train_dataset = datasets.ImageFolder(
            '../data/tiny-imagenet-200/train/',
            transforms.Compose([
                transforms.RandomCrop(64, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        test_dataset = datasets.ImageFolder(
            '../data/tiny-imagenet-200/val/',
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

    return train_dataset, test_dataset, num_classes


def get_data(name='cifar10', train_bs=128, test_bs=1000):
    """
        return: train_data_loader, test_data_loader
    """
    train_dataset, test_dataset = get_dataset(name)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_bs, shuffle=False)
    return train_loader, test_loader