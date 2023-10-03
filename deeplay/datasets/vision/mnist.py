import torch
import torchvision
import functools


class MNIST(torchvision.datasets.MNIST):
    """MNIST Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.'
        backend (string): 'torch' or None. Default: 'torch'. 
            Converts the output to the specified backend. If None, returns the output as is.
    """


    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, backend='pytorch'):
        super().__init__(root, train, transform, target_transform, download)
        self.backend = backend

    def __getitem__(self, index):
        
        img, target = super().__getitem__(index)
        
        if self.backend == 'torch' or self.backend == 'pytorch':
            img = torchvision.transforms.ToTensor()(img)
            target = torch.tensor(target)
              
        return img, target