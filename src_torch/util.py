# coding: utf-8

import numpy as np
import torch

from torchvision import datasets, models, transforms

from dataset import CelebA_attr, GTSRB, CustomGTSRBDataSet, CustomMNISTMDataSet, get_custom_folder
#from network import resnet18
from resnet_cifar import resnet18, resnet50
from vgg_cifar import vgg11_bn
from mobilenetv2 import MobileNetV2
from densenet import densenet
from mobilenet import MobileNet
from shufflenetv2 import shufflenetv2

_mean = {
    'default':  [0.5   , 0.5   , 0.5   ],
    'mnist':    [0.5   , 0.5   , 0.5   ],
    'mnistm':    [0.5   , 0.5   , 0.5   ],
    'fmnist':    [0.5],
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'gtsrb':    [0.0   , 0.0   , 0.0   ],
    'celeba':   [0.0   , 0.0   , 0.0   ],
    'imagenet': [0.485 , 0.456 , 0.406 ],
    'caltech': [0.485 , 0.456 , 0.406 ],
    'asl': [0.485 , 0.456 , 0.406 ],
}

_std = {
    'default':  [0.5   , 0.5   , 0.5   ],
    'mnist':    [0.5   , 0.5   , 0.5   ],
    'mnistm':    [0.5   , 0.5   , 0.5   ],
    'fmnist':    [0.5],
    'cifar10':  [0.2471, 0.2435, 0.2616],
    'gtsrb':    [1.0   , 1.0   , 1.0   ],
    'celeba':   [1.0   , 1.0   , 1.0   ],
    'imagenet': [0.229 , 0.224 , 0.225 ],
    'caltech': [0.229 , 0.224 , 0.225 ],
    'asl': [0.229 , 0.224 , 0.225 ],
}

_size = {
    'fmnist':    ( 28,  28, 1),
    'mnist':    ( 28,  28, 1),
    'mnistm':  ( 32,  32, 3),
    'cifar10':  ( 32,  32, 3),
    'gtsrb':    ( 32,  32, 3),
    'celeba':   ( 64,  64, 3),
    'imagenet': (224, 224, 3),
    'caltech': (224, 224, 3),
    'asl': (200, 200, 3),
}

_num = {
    'fmnist':    10,
    'mnist':    10,
    'mnistm':    10,
    'cifar10':  10,
    'gtsrb':    43,
    'celeba':   8,
    'imagenet': 1000,
    'asl': 29,
    'caltech': 101,
}


def get_norm(dataset):
    mean, std = _mean[dataset], _std[dataset]
    mean_t = torch.Tensor(mean)
    std_t  = torch.Tensor( std)
    return mean_t, std_t


def preprocess(x, dataset, clone=True, channel_first=True):
    if torch.is_tensor(x):
        x_out = torch.clone(x) if clone else x
    else:
        x_out = torch.FloatTensor(x)

    if x_out.max() > 100:
        x_out = x_out / 255.

    if channel_first:
        x_out = x_out.permute(0, 2, 3, 1)

    mean_t, std_t = get_norm(dataset)
    mean_t = mean_t.to(x_out.device)
    std_t  = std_t.to(x_out.device)

    x_out = (x_out - mean_t) / std_t
    x_out = x_out.permute(0, 3, 1, 2)
    return x_out


def deprocess(x, dataset, clone=True):
    mean_t, std_t = get_norm(dataset)
    mean_t = mean_t.to(x.device)
    std_t  = std_t.to(x.device)

    x_out = torch.clone(x) if clone else x
    x_out = x_out.permute(0, 2, 3, 1) * std_t + mean_t
    x_out = x_out.permute(0, 3, 1, 2)
    return x_out


def get_num(dataset):
    return _num[dataset]


def get_size(dataset):
    return _size[dataset]


def get_dataloader(dataset, train=True, ratio=1.0, batch_size=128):
    transforms_list = []
    #transforms_list.append(transforms.Resize(_size[dataset][:2]))
    transforms_list.append(transforms.ToTensor())
    #transforms_list.append(transforms.Normalize(_mean[dataset], _std[dataset]))
    transform = transforms.Compose(transforms_list)

    data_root = './data'
    if dataset == 'gtsrb':
        dataset = CustomGTSRBDataSet(data_root, train, transform)
    elif dataset == 'mnist':
        dataset = datasets.MNIST(data_root, train, transform, download=True)
    elif dataset == 'fmnist':
        dataset = datasets.FashionMNIST(data_root, train, transform, download=True)
    elif dataset == 'cifar10':
        dataset = datasets.CIFAR10(data_root, train, transform, download=True)
    elif dataset == 'celeba':
        dataset = CelebA_attr(data_root, train, transform)
    elif dataset == 'mnistm':
        dataset = CustomMNISTMDataSet(data_root, train, transform)
    elif dataset == 'asl':
        dataset = get_custom_folder(data_root + '/' + str(dataset), train, transform)
    elif dataset == 'caltech':
        my_trans = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = get_custom_folder(data_root + '/' + str(dataset), train, my_trans)
    else:
        raise Exception('Invalid dataset')
    if ratio < 1:
        indices = np.arange(int(len(dataset) * ratio))
        dataset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             num_workers=6, shuffle=train)
    return dataloader


def get_model(args):
    if args.dataset == 'imagenet':
        if args.model == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif args.model == 'resnet50':
            model = models.resnet50(pretrained=True)
    if args.model == 'resnet18':
        model = resnet18()
    elif args.model == 'resnet50':
        model = resnet50()
    elif args.model == 'vgg11_bn':
        if args.dataset == 'gtsrb':
            model = vgg11_bn(num_classes=43)
    elif args.model == 'MobileNetV2':
        model = MobileNetV2()
    elif args.model == 'MobileNet':
        model = MobileNet(num_classes=29)
    elif args.model == 'densenet':
        model = densenet(num_classes=10)
    elif args.model == 'shufflenetv2':
        model = shufflenetv2(num_classes=101)
    return model


def pgd_attack(model, images, labels, mean, std,
               eps=0.3, alpha=2/255, iters=40):
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    images = images + 2 * (torch.rand_like(images) - 0.5) * eps
    images = torch.clamp(images, 0, 1)

    mean = mean.to(images.device)
    std  = std.to(images.device)

    for i in range(iters):
        images.requires_grad = True

        outputs = model(
                    ((images.permute(0, 2, 3, 1) - mean) / std)\
                            .permute(0, 3, 1, 2)
                  )

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images
