#!/usr/bin/env python3
"""
Experiment runner script for grid search over batch_size, lr_opt, and alpha.
Runs main_opt.py with different parameter combinations and aggregates results.
Each configuration is run multiple times to get average metrics.
"""

import subprocess
import os
import sys
import re
import csv
import shutil
from datetime import datetime
import logging
import glob
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define hyperparameter grids
BATCH_SIZES = [1, 2, 4, 8, 16, 24]
LEARNING_RATES = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
ALPHA_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0]

# Number of times to run each configuration
NUM_RUNS = 5

# Other fixed arguments
ITERATIONS = 800
DATASET = 'mnist'
PRETRAINED = False
DEVICE = 'cuda'

def run_experiment(batch_size, lr_opt, alpha, exp_id, run_num, exp_dir):
    """
    Run a single experiment with given hyperparameters.
    Returns a dict with metrics extracted from the log.
    exp_dir: directory to save viz files for this configuration
    """
    logger.info(f"Running experiment {exp_id}, run {run_num}/{NUM_RUNS}: BS={batch_size}, LR={lr_opt}, Alpha={alpha}")
    
    cmd = [
        'python', 'main_opt.py',
        '--batch_size', str(batch_size),
        '--lr_opt', str(lr_opt),
        '--alpha', str(alpha),
        '--iterations', str(ITERATIONS),
        '--dataset', DATASET,
    ]
    
    if PRETRAINED:
        cmd.append('--pretrained')
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            logger.error(f"Experiment {exp_id}, run {run_num} failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr}")
            return None
        
        # Parse metrics from stdout and stderr
        output = result.stdout + result.stderr
        metrics = parse_metrics(output)
        
        if metrics is None:
            logger.warning(f"Could not parse metrics from experiment {exp_id}, run {run_num}")
            return None
        
        # Move viz file to experiment directory
        viz_file = get_latest_viz_file()
        if viz_file:
            src_path = os.path.join('./viz', viz_file)
            dst_path = os.path.join(exp_dir, f'run_{run_num}_{viz_file}')
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                logger.info(f"Saved viz to {dst_path}")
                metrics['viz_file'] = f'run_{run_num}_{viz_file}'
        
        logger.info(f"Experiment {exp_id}, run {run_num} completed. PSNR: {metrics.get('psnr_mean', 'N/A'):.2f}, SSIM: {metrics.get('ssim_mean', 'N/A'):.4f}")
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"Experiment {exp_id}, run {run_num} timed out")
        return None
    except Exception as e:
        logger.error(f"Experiment {exp_id}, run {run_num} failed with exception: {e}")
        return None

def parse_metrics(output):
    """
    Extract PSNR, SSIM, MSE from log output.
    """
    metrics = {}
    
    # Pattern: "Final Metrics (first N) - PSNR mean: 12.34, SSIM mean: 0.5678, MSE: 0.001234"
    pattern = r'Final Metrics.*?PSNR mean:\s*([\d.]+),\s*SSIM mean:\s*([\d.]+),\s*MSE:\s*([\d.]+)'
    match = re.search(pattern, output)
    
    if match:
        metrics['psnr_mean'] = float(match.group(1))
        metrics['ssim_mean'] = float(match.group(2))
        metrics['mse'] = float(match.group(3))
        return metrics
    
    logger.warning("Could not find metrics pattern in output")
    return None

def get_latest_viz_file():
    """
    Get the most recently created viz file.
    """
    viz_dir = './viz'
    if not os.path.exists(viz_dir):
        return None
    
    files = [f for f in os.listdir(viz_dir) if f.startswith('reconstruction_opt_') and f.endswith('.png')]
    if not files:
        return None
    
    files.sort()
    return files[-1]

def organize_results(all_results):
    """
    Create summary table with averaged metrics across runs.
    """
    summary_dir = './results_summary'
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    # Group results by hyperparameter configuration
    config_results = {}
    for result in all_results:
        if result is None:
            continue
        
        key = (result['batch_size'], result['lr_opt'], result['alpha'])
        if key not in config_results:
            config_results[key] = []
        config_results[key].append(result)
    
    # Create summary CSV with averaged metrics
    csv_file = os.path.join(summary_dir, 'results_summary.csv')
    fieldnames = ['batch_size', 'lr_opt', 'alpha', 'psnr_mean_avg', 'psnr_std', 'ssim_mean_avg', 'ssim_std', 'mse_avg', 'mse_std', 'num_runs']
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for (batch_size, lr_opt, alpha), results in sorted(config_results.items()):
            psnr_vals = [r.get('psnr_mean', 0) for r in results]
            ssim_vals = [r.get('ssim_mean', 0) for r in results]
            mse_vals = [r.get('mse', 0) for r in results]
            
            psnr_mean = np.mean(psnr_vals)
            psnr_std = np.std(psnr_vals)
            ssim_mean = np.mean(ssim_vals)
            ssim_std = np.std(ssim_vals)
            mse_mean = np.mean(mse_vals)
            mse_std = np.std(mse_vals)
            
            writer.writerow({
                'batch_size': batch_size,
                'lr_opt': lr_opt,
                'alpha': alpha,
                'psnr_mean_avg': f"{psnr_mean:.2f}",
                'psnr_std': f"{psnr_std:.2f}",
                'ssim_mean_avg': f"{ssim_mean:.4f}",
                'ssim_std': f"{ssim_std:.4f}",
                'mse_avg': f"{mse_mean:.6f}",
                'mse_std': f"{mse_std:.6f}",
                'num_runs': len(results)
            })
    
    logger.info(f"Summary CSV saved to {csv_file}")
    
    # Print summary table
    print("\n" + "="*120)
    print("EXPERIMENT SUMMARY (Averaged across {} runs)".format(NUM_RUNS))
    print("="*120)
    print(f"{'BS':<4} {'LR':<10} {'Alpha':<8} {'PSNR Mean':<15} {'PSNR Std':<10} {'SSIM Mean':<15} {'SSIM Std':<10} {'MSE':<15}")
    print("-"*120)
    
    for (batch_size, lr_opt, alpha), results in sorted(config_results.items()):
        psnr_vals = [r.get('psnr_mean', 0) for r in results]
        ssim_vals = [r.get('ssim_mean', 0) for r in results]
        mse_vals = [r.get('mse', 0) for r in results]
        
        psnr_mean = np.mean(psnr_vals)
        psnr_std = np.std(psnr_vals)
        ssim_mean = np.mean(ssim_vals)
        ssim_std = np.std(ssim_vals)
        mse_mean = np.mean(mse_vals)
        
        print(f"{batch_size:<4} {lr_opt:<10.0e} {alpha:<8.3f} {psnr_mean:<15.2f} {psnr_std:<10.2f} {ssim_mean:<15.4f} {ssim_std:<10.4f} {mse_mean:<15.6f}")
    
    print("="*120 + "\n")

def main():
    """
    Main function: run grid search multiple times and organize results.
    """
    logger.info(f"Starting experiment grid search")
    logger.info(f"Batch sizes: {BATCH_SIZES}")
    logger.info(f"Learning rates: {LEARNING_RATES}")
    logger.info(f"Alpha values: {ALPHA_VALUES}")
    logger.info(f"Number of runs per configuration: {NUM_RUNS}")
    logger.info(f"Total experiments: {len(BATCH_SIZES) * len(LEARNING_RATES) * len(ALPHA_VALUES) * NUM_RUNS}")
    
    all_results = []
    exp_id = 1
    
    for batch_size in BATCH_SIZES:
        for lr_opt in LEARNING_RATES:
            for alpha in ALPHA_VALUES:
                # Create experiment-specific directory
                exp_dir_name = f"bs{batch_size}_lr{lr_opt:.0e}_alpha{alpha}"
                exp_dir = os.path.join('./viz', exp_dir_name)
                os.makedirs(exp_dir, exist_ok=True)
                logger.info(f"Created experiment directory: {exp_dir}")
                
                # Run the experiment NUM_RUNS times
                for run_num in range(1, NUM_RUNS + 1):
                    result = run_experiment(batch_size, lr_opt, alpha, exp_id, run_num, exp_dir)
                    if result is not None:
                        result['exp_id'] = exp_id
                        result['batch_size'] = batch_size
                        result['lr_opt'] = lr_opt
                        result['alpha'] = alpha
                    all_results.append(result)
                
                exp_id += 1
    
    # Organize and summarize results
    organize_results(all_results)
    
    logger.info("Experiment grid search completed!")

if __name__ == '__main__':
    main()
