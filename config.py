import torch


DATASETS = ["cifar10", "cifar100", "stl10", "tiny_imagenet", "imagenet"]
DEFAULT_DATASET = "cifar10"
DEFAULT_NUM_CLASSES = 10
DEFAULT_MODEL = "vit_tiny_patch16_224"
USE_CUSTOM_VIT = True
DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEFAULT_SAVE_PATH = "checkpoints"
DEFAULT_LOG_PATH = "logs"
