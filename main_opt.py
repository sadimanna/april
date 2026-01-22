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
    parser = argparse.ArgumentParser(description='APRIL')
    parser.add_argument('--lora_rank', type=int, default=0, help='LoRA rank (default: 4). Set to 0 to disable.')
    parser.add_argument('--use_layernorm', action='store_true', help='Use LayerNorm in the first block')
    parser.add_argument('--use_residual', action='store_true', help='Use Residual connection in the first block')
    parser.add_argument('--iterations', type=int, default=1500, help='Number of optimization iterations')
    parser.add_argument('--lr_opt', type=float, default=1e-3, help='Learning rate for input optimization')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--dataset', type=str, default=config.DEFAULT_DATASET, help='Dataset to use (default from config)')
    args = parser.parse_args()

    # Print Banner
    print_banner(args.lora_rank)

    # Log hyperparameters
    logging.info("================ Hyperparameters ================")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("================================================")

    # 1. Load the dataset
    dataloader = get_dataloader(args.dataset, root='./data', split='test')

    # 2. Create the model
    if config.USE_CUSTOM_VIT:
        # Automatic config selection
        if args.dataset in ['cifar10', 'cifar100']:
             model_name = 'vit_small_cifar'
             logging.info(f"Dataset is {args.dataset}, using {model_name}")
        elif args.dataset == 'mnist':
             model_name = 'vit_small_mnist'
             logging.info(f"Dataset is {args.dataset}, using {model_name}")
        elif args.dataset == 'stl10':
             model_name = 'vit_medium_stl10'
             logging.info(f"Dataset is {args.dataset}, using {model_name}")
        elif 'imagenet' in args.dataset:
             model_name = 'vit_base_patch16_224'
             logging.info(f"Dataset is {args.dataset}, using {model_name}")
        else:
             model_name = config.DEFAULT_MODEL
             logging.info(f"Dataset {args.dataset} not in auto-select list, using default {model_name}")

        logging.info(f"Model Options: LoRA Rank={args.lora_rank}, LN={args.use_layernorm}, Res={args.use_residual}")

        model = get_custom_vit(model_name,
                                 pretrained=args.pretrained,
                                 num_classes=config.DEFAULT_NUM_CLASSES,
                                 use_layernorm=args.use_layernorm, 
                                 use_residual=args.use_residual,
                                 lora_rank=args.lora_rank,
                                 patch_embed_type='linear')
    else:
        model = get_model(config.DEFAULT_MODEL,
                              pretrained=args.pretrained,
                              num_classes=config.DEFAULT_NUM_CLASSES)
    
    model.to(config.DEFAULT_DEVICE)
    model.eval() # Usually eval mode is used for reconstruction attacks

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total Parameters: {total_params:,}")
    logging.info(f"Trainable Parameters (Initial): {trainable_params:,}")

    # 3. Training Loop Setup
    criterion = nn.CrossEntropyLoss()
    
    # a. Pass a single sample through the model
    images, labels = next(iter(dataloader))
    images = images.to(config.DEFAULT_DEVICE)
    labels = labels.to(config.DEFAULT_DEVICE)
    
    # b. Calculate ground truth gradients
    # Enable gradients for relevant parameters
    params_to_track = []
    for n, p in model.named_parameters():
        if 'lora' in n.lower() or 'head' in n.lower() or 'pos_embed' in n.lower():
            p.requires_grad = True
            params_to_track.append(p)
    
    logging.info(f"Number of parameters tracked for reconstruction: {len(params_to_track)}")
    tracked_param_count = sum(p.numel() for p in params_to_track)
    logging.info(f"Total elements in tracked parameters: {tracked_param_count:,}")
    
    output_gt = model(images)
    loss_gt = criterion(output_gt, labels)
    grads_gt = torch.autograd.grad(loss_gt, params_to_track, create_graph=False)
    grads_gt = [g.detach() for g in grads_gt]
    
    gt_grad_norms = [g.norm().item() for g in grads_gt]
    logging.info(f"Ground Truth Gradients computed. Mean norm: {np.mean(gt_grad_norms):.6f}, Max norm: {np.max(gt_grad_norms):.6f}")
    
    # Positional embedding gradient specifically (often the last or easily identifiable)
    pos_embed_grad_gt = None
    for n, g in zip([n for n, p in model.named_parameters() if p.requires_grad], grads_gt):
        if 'pos_embed' in n.lower():
            pos_embed_grad_gt = g
            break

    # d. Randomly initialize sample x_opt
    x_opt = torch.randn_like(images).to(config.DEFAULT_DEVICE).requires_grad_(True)
    
    # e-g. Optimization loop
    optimizer_x = torch.optim.Adam([x_opt], lr=args.lr_opt)
    
    num_iterations = args.iterations
    logging.info(f"Starting optimization for {num_iterations} iterations...")
    
    for i in range(num_iterations):
        optimizer_x.zero_grad()
        model.zero_grad()
        
        output_opt = model(x_opt)
        loss_opt = criterion(output_opt, labels)
        
        # Calculate gradients for x_opt
        grads_opt = torch.autograd.grad(loss_opt, params_to_track, create_graph=True)
        
        # Gradient matching loss (MSE)
        grad_loss = 0
        for g_o, g_g in zip(grads_opt, grads_gt):
            grad_loss += torch.nn.functional.mse_loss(g_o, g_g)
            
        # Positional embedding gradient loss (MSE)
        pos_loss = 0
        if pos_embed_grad_gt is not None:
            # Find pos_embed grad in grads_opt
            pos_embed_grad_opt = None
            for n, g in zip([n for n, p in model.named_parameters() if p.requires_grad], grads_opt):
                if 'pos_embed' in n.lower():
                    pos_embed_grad_opt = g
                    break
            if pos_embed_grad_opt is not None:
                pos_loss = 1 - torch.nn.functional.cosine_similarity(pos_embed_grad_opt.flatten(), pos_embed_grad_gt.flatten(), dim=0)

        total_loss = grad_loss + pos_loss
        total_loss.backward()
        optimizer_x.step()
        
        if (i+1) % 50 == 0 or i == 0:
            x_opt_norm = torch.norm(x_opt).item()
            logging.info(f"Iteration {i+1}/{num_iterations}, Total Loss: {total_loss.item():.6f}, Grad Loss: {grad_loss.item():.6f}, Pos Loss: {pos_loss:.6f}, x_opt Norm: {x_opt_norm:.4f}")

    # 5. Metrics and Visualization
    logging.info("Optimization finished. Calculating metrics...")
    
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric

    # Convert to numpy for skimage
    x_gt_np = images[0].detach().cpu().permute(1, 2, 0).numpy()
    x_opt_np = x_opt[0].detach().cpu().permute(1, 2, 0).numpy()
    
    # Un-normalize if needed (assuming standard normalization)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    x_gt_np = np.clip(std * x_gt_np + mean, 0, 1)
    x_opt_np = np.clip(std * x_opt_np + mean, 0, 1)
    
    psnr_val = psnr_metric(x_gt_np, x_opt_np, data_range=1.0)
    # Use win_size=3 or similar for small images like CIFAR
    ssim_val = ssim_metric(x_gt_np, x_opt_np, data_range=1.0, channel_axis=2, win_size=3)
    mse_val = np.mean((x_gt_np - x_opt_np) ** 2)
    
    logging.info(f"Final Metrics - PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}, MSE: {mse_val:.6f}")
    
    # Save results
    viz_dir = './viz'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(x_gt_np)
    axes[0].set_title("Original (Ground Truth)")
    axes[0].axis('off')
    
    axes[1].imshow(x_opt_np)
    axes[1].set_title(f"Reconstructed (PSNR: {psnr_val:.2f})")
    axes[1].axis('off')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(viz_dir, f'reconstruction_opt_{timestamp}.png'))
    logging.info(f"Saved reconstruction result to {viz_dir}/reconstruction_opt_{timestamp}.png")
