
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.custom_vit import get_custom_vit

def test_lora():
    print("Testing LoRA Implementation...")
    
    # 1. Instantiate model with LoRA
    rank = 4
    model = get_custom_vit(model_name='vit_base_patch16_224', lora_rank=rank)
    print(f"Model instantiated with LoRA rank {rank}")
    
    # 2. Check for LoRA parameters
    lora_params = [n for n, p in model.named_parameters() if 'lora' in n]
    assert len(lora_params) > 0, "No LoRA parameters found!"
    print(f"Found {len(lora_params)} LoRA parameters parameters")
    
    # 3. Check lora_B initialization (should be zero)
    for name, param in model.named_parameters():
        if 'lora_B' in name:
            assert torch.all(param == 0), f"{name} is not initialized to zero!"
    print("All lora_B parameters initialized to zero.")
    
    # 4. Forward pass comparison (lora vs no lora)
    # Since lora_B is zero, output should be identical to non-LoRA output (or close to it dependent on other factors)
    # But wait, we replaced the Attention class. We should compare against a model with lora_rank=0.
    
    model_no_lora = get_custom_vit(model_name='vit_base_patch16_224', lora_rank=0)
    
    # Copy weights from lora model to no-lora model (excluding lora weights)
    state_dict = model.state_dict()
    state_dict_no_lora = {k: v for k, v in state_dict.items() if 'lora' not in k}
    
    # We need to be careful. The Attention classes are 'compatible' in state dict if keys match.
    # CustomAttention with rank=0 should have same keys as Timm Attention?
    # Actually, CustomAttention w/ rank=0 only has qkv, proj. Timm Attention has same.
    # The structure of CustomAttention matches Timm's Attention enough for weight loading? 
    # Yes, we designed it that way.
    
    model_no_lora.load_state_dict(state_dict_no_lora, strict=True)
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    with torch.no_grad():
        out_lora = model(dummy_input)
        out_no_lora = model_no_lora(dummy_input)
        
    diff = torch.norm(out_lora - out_no_lora)
    print(f"Difference between LoRA (init) and No-LoRA outputs: {diff.item()}")
    assert diff < 1e-6, "Outputs match failed!"
    print("Outputs match confirmed.")
    
    # 5. Backward pass check
    # Make sure lora parameters get gradients
    out = model(dummy_input)
    loss = out.sum()
    loss.backward()
    
    grad_found = False
    for name, param in model.named_parameters():
        if 'lora' in name:
            if param.grad is not None:
                grad_found = True
                # print(f"Gradient found for {name}")
            else:
                print(f"No gradient for {name}")
                
    assert grad_found, "No gradients found for LoRA parameters!"
    print("Gradients flow to LoRA parameters confirmed.")
    
    print("\nTest Passed Successfully!")

if __name__ == "__main__":
    test_lora()
