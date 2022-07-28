import os
import torch
import random
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import torchvision.datasets as datasets

def datasetload(in_datasets, padding_mode='constant', dataset_dir=None):
    print(in_datasets)
    if in_datasets == 'imagenet':
        if dataset_dir is None:
            datapath = os.path.expanduser('/home/Databases/ILSVRC2012/')
        else:
            datapath = os.path.expanduser(str(dataset_dir))

        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        jittering = utils_imagenet.ColorJitter(brightness=0.4, contrast=0.4,
                                               saturation=0.4)
        lighting = utils_imagenet.Lighting(alphastd=0.1,
                                           eigval=[0.2175, 0.0188, 0.0045],
                                           eigvec=[[-0.5675, 0.7192, 0.4009],
                                                   [-0.5808, -0.0045, -0.8140],
                                                   [-0.5836, -0.6948, 0.4203]])

        train_dataset = datasets.ImageFolder(
                                            traindir,
                                            transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                jittering,
                                                lighting,
                                                normalize,
                                            ]))
        val_dataset = datasets.ImageFolder(valdir,
                                           transforms.Compose([transforms.Resize(256),
                                                               transforms.CenterCrop(224),
                                                               transforms.ToTensor(),
                                                               normalize,
                                                               ]))

    elif in_datasets == 'tiny_imagenet':
        if dataset_dir is None:
            datapath = os.path.expanduser('/home/Databases/tiny-imagenet-200')
        else:
            datapath = os.path.expanduser(str(dataset_dir))
        traindir = os.path.join(datapath, 'train')
        valdir = os.path.join(datapath, 'val')
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
        normalize = transforms.Normalize(mean, std)
        train_dataset = datasets.ImageFolder(traindir,
            transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(64, padding=4,
                                        padding_mode = padding_mode),
                                transforms.ToTensor(), normalize,
                                ]))
        val_dataset = datasets.ImageFolder(valdir,
            transforms.Compose([
                                transforms.ToTensor(), normalize,
                                ]))

    elif in_datasets == 'cifar10' or 'cifar100':
        if dataset_dir is None:
            datapath = os.path.expanduser('/home/Databases/cifar/cifar-100-python')
        else:
            datapath = os.path.expanduser(str(dataset_dir))
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        if in_datasets == 'cifar10':
            train_dataset = CIFAR10(root=datapath, train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        	transforms.RandomCrop(32, padding=4,
                                                    padding_mode = padding_mode),
                                        	transforms.RandomHorizontalFlip(),
                                        	transforms.ToTensor(),
                                        	normalize,
                                            ]))
            val_dataset   = CIFAR10(root=datapath, train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                            ]))
        else:
            train_dataset = CIFAR100(root=datapath, train=True,
                                    download=True,
                                    transform=transforms.Compose([
                                        	transforms.RandomCrop(32, padding=4,
                                                    padding_mode = padding_mode),
                                        	transforms.RandomHorizontalFlip(),
                                        	transforms.ToTensor(),
                                        	normalize,
                                            ]))
            val_dataset   = CIFAR100(root=datapath, train=False,
                                    download=True,
                                    transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize,
                                            ]))
    elif in_datasets == 'mnist':
        datapath = os.path.expanduser('/home/Databases/mnist/')
        print(datapath)
        train_dataset = MNIST(datapath, train=True, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))]))

        val_dataset = MNIST(datapath, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))]))
    else:
        raise NotImplementedError

    return train_dataset, val_dataset


## For ImageNet
class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class ColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        self.transforms = []
        if self.brightness != 0:
            self.transforms.append(Brightness(self.brightness))
        if self.contrast != 0:
            self.transforms.append(Contrast(self.contrast))
        if self.saturation != 0:
            self.transforms.append(Saturation(self.saturation))

        random.shuffle(self.transforms)
        transform = Compose(self.transforms)
        # print(transform)
        return transform(img)
