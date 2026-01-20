
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.custom_vit import get_custom_vit
from models.model import April

def test_lora_gradients():
    print("Testing LoRA Gradient Extraction...")
    
    # 1. Instantiate model with LoRA
    rank = 4
    vit = get_custom_vit(model_name='vit_base_patch16_224', lora_rank=rank)
    model = April(vit)
    
    # Perturb lora_B to avoid zero gradients (singularity at init)
    with torch.no_grad():
        for name, param in model.named_parameters():
             if 'lora_B' in name:
                 param.add_(torch.randn_like(param) * 0.01)
                 
    # 2. Run Forward and Backward
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    
    # 3. Get Gradients
    block_index = 0
    dldq, dldk, dldv = model.get_attention_gradients(block_index)
    
    # Check shapes
    embed_dim = 768
    expected_shape = (embed_dim, embed_dim)
    
    print(f"dldq shape: {dldq.shape}")
    assert dldq.shape == expected_shape, f"dldq shape mismatch: expected {expected_shape}, got {dldq.shape}"
    assert dldk.shape == expected_shape, "dldk shape mismatch"
    assert dldv.shape == expected_shape, "dldv shape mismatch"
    
    # Check if not zero (assuming random weights lead to non-zero grad)
    print(f"dldq norm: {torch.norm(dldq)}")
    assert torch.norm(dldq) > 0, "Gradient is zero!"
    
    # Check LoRA weights
    q, k, v = model.get_attention_weights(block_index)
    print(f"q weights shape: {q.shape}")
    assert q.shape == expected_shape, "Weight shape mismatch"
    
    print("\nLoRA Gradient Extraction Verified!")

if __name__ == "__main__":
    test_lora_gradients()
