from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os

class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.dataset = self._get_dataset()

    def _get_dataset(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

class CIFAR10Dataset(ImageDataset):
    def _get_dataset(self):
        return datasets.CIFAR10(root=self.root, train=True, download=True)

class CIFAR100Dataset(ImageDataset):
    def _get_dataset(self):
        return datasets.CIFAR100(root=self.root, train=True, download=True)

class STL10Dataset(ImageDataset):
    def _get_dataset(self):
        return datasets.STL10(root=self.root, split='train', download=True)

class TinyImageNetDataset(ImageDataset):
    def _get_dataset(self):
        # Assuming Tiny ImageNet is structured like ImageFolder
        return datasets.ImageFolder(root=os.path.join(self.root, 'tiny-imagenet-200', 'train'))

class ImageNetDataset(ImageDataset):
    def _get_dataset(self):
        # Assuming ImageNet is structured like ImageFolder
        return datasets.ImageFolder(root=os.path.join(self.root, 'train'))

def get_dataset(dataset_name, root, transform=None):
    if dataset_name == "cifar10":
        return CIFAR10Dataset(root=root, transform=transform)
    elif dataset_name == "cifar100":
        return CIFAR100Dataset(root=root, transform=transform)
    elif dataset_name == "stl10":
        return STL10Dataset(root=root, transform=transform)
    elif dataset_name == "tiny_imagenet":
        return TinyImageNetDataset(root=root, transform=transform)
    elif dataset_name == "imagenet":
        return ImageNetDataset(root=root, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
