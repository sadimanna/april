import torch
import numpy as np
import config
from dataloaders import get_dataloader
from models.model import get_model, April
from models.custom_vit import get_custom_vit
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import logging
import os
import sys
from scipy.ndimage import median_filter

def setup_logging():
    # Configure logging
    log_dir = config.DEFAULT_LOG_PATH
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'april_run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

import argparse
from banner import print_banner

if __name__ == '__main__':
    # Initialize logging
    setup_logging()

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='APRIL with LoRA')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank (default: 4). Set to 0 to disable.')
    parser.add_argument('--use_layernorm', action='store_true', help='Use LayerNorm in the first block')
    parser.add_argument('--use_residual', action='store_true', help='Use Residual connection in the first block')
    args = parser.parse_args()

    # Print Banner
    print_banner(args.lora_rank)

    # 1. Load the dataset
    dataloader = get_dataloader(config.DEFAULT_DATASET, root='./data')

    # 2. Create the model
    if config.USE_CUSTOM_VIT:
        
        # Automatic config selection
        if config.DEFAULT_DATASET in ['cifar10', 'cifar100']:
             model_name = 'vit_small_cifar'
             logging.info(f"Dataset is {config.DEFAULT_DATASET}, using {model_name}")
        elif config.DEFAULT_DATASET == 'mnist':
             model_name = 'vit_small_mnist'
             logging.info(f"Dataset is {config.DEFAULT_DATASET}, using {model_name}")
        elif config.DEFAULT_DATASET == 'stl10':
             model_name = 'vit_medium_stl10'
             logging.info(f"Dataset is {config.DEFAULT_DATASET}, using {model_name}")
        elif 'imagenet' in config.DEFAULT_DATASET:
             model_name = 'vit_base_patch16_224'
             logging.info(f"Dataset is {config.DEFAULT_DATASET}, using {model_name}")
        else:
             model_name = config.DEFAULT_MODEL
             logging.info(f"Dataset {config.DEFAULT_DATASET} not in auto-select list, using default {model_name}")

        logging.info(f"Model Options: LoRA Rank={args.lora_rank}, LN={args.use_layernorm}, Res={args.use_residual}")

        vit_model = get_custom_vit(model_name,
                                 pretrained=False,
                                 num_classes=config.DEFAULT_NUM_CLASSES,
                                 use_layernorm=args.use_layernorm, 
                                 use_residual=args.use_residual,
                                 lora_rank=args.lora_rank,
                                 patch_embed_type='linear')
    else:
        vit_model = get_model(config.DEFAULT_MODEL,
                              pretrained=False,
                              num_classes=config.DEFAULT_NUM_CLASSES)
    

