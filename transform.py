import numpy as np
import torch
import torchvision
from torch import nn
from torchvision import transforms
import PIL

class MoCoTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size=32, gaussian=False):
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(mode='RGB'),
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()])

        self.test_transform = transforms.Compose([
            torchvision.transforms.Resize(size=size),
            transforms.ToTensor()]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class SimSiamTransform:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size=32, gaussian=False):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        if gaussian:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='RGB'),
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    GaussianBlur(kernel_size=int(0.1 * size)),
                    torchvision.transforms.ToTensor(),
                ]
            )
        else:
            self.train_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(mode='RGB'),
                    torchvision.transforms.RandomResizedCrop(size=size),
                    torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
                    torchvision.transforms.RandomApply([color_jitter], p=0.8),
                    torchvision.transforms.RandomGrayscale(p=0.2),
                    torchvision.transforms.ToTensor(),
                ]
            )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)



class SimCLRTransform:
    def __init__(self, size=32, gaussian=False, data_format="array"):
        s = 1
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        self.gaussian = gaussian
        self.size = size
        self.data_format = data_format

        self.base_transform = [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]
        
        self.no_augment = [transforms.RandomResizedCrop(size=size), transforms.ToTensor()]
        
        if gaussian:
            self.base_transform.append(GaussianBlur(kernel_size=int(0.1 * size)))
            
        self.base_transform.append(transforms.ToTensor())

    def __call__(self, x):
        if isinstance(x, PIL.Image.Image):
            transform_list = self.base_transform
            original = self.no_augment
        else:
            transform_list = [transforms.ToPILImage()] + self.base_transform
            no_augment =  [transforms.ToPILImage()] + self.no_augment 
        
        transform = transforms.Compose(transform_list)
        no_augment = transforms.Compose(original)
        
        
        return no_augment(x), transform(x), transform(x)

    def test_transform(self):
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])

    def fine_tune_transform(self):
        return transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

