import torch
import torchvision.transforms as transforms
import torchvision
import numpy as np


def load_data(batch_size, device_ids):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size * len(device_ids),
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size * len(device_ids),
                                             shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def manifold_mixup_data(x, y, alpha):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""

    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)  # .cuda()
#    print(index)
    mixed_x = lam * x + (1 - lam) * x[index, :]
#    print(x[index, :])
    y_a, y_b = y, y[index]
    print(y_a, y_b)
    return mixed_x, y_a, y_b, lam, index


def manifold_mixup_data_demo(x1, x2, alpha):
    """mixup demo for mini_batch_size=1"""
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    mixed_x = lam * x1 + (1 - lam) * x2
    return mixed_x, lam


def manifold_mixup_weight(weight1, weight2, lam, *args ):
    mix_weight = lam * weight1 + (1 - lam) * weight2
    return mix_weight


def mixup_criterion(y_a, y_b, lambd):
    return lambda criterion, pred: lambd * criterion(pred, y_a) + (1 - lambd) * criterion(pred, y_b)
