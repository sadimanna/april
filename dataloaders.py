from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import get_dataset
import config

def get_dataloader(dataset_name, root, split='train', batch_size=config.DEFAULT_BATCH_SIZE, num_workers=4):
    """
    Creates a DataLoader for the specified dataset.
    """
    if dataset_name in ["cifar10", "cifar100"]:
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif dataset_name == "stl10":
        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:  # tiny_imagenet, imagenet
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = get_dataset(dataset_name, root, split=split, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
