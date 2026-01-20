
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.custom_vit import get_custom_vit
from models.vit_config import VIT_CONFIGS

def test_configs():
    print("Testing ViT Configurations...")

    configs_to_test = [
        ('vit_small_cifar', 384, 4, 4, 16),
        ('vit_small_mnist', 384, 4, 4, 14),
        ('vit_medium_stl10', 384, 6, 6, 16),
    ]

    for name, embed_dim, depth, num_heads, patch_size in configs_to_test:
        print(f"\nTesting {name}...")
        try:
            model = get_custom_vit(model_name=name)
            
            # Check Embed Dim
            assert model.embed_dim == embed_dim, f"Embed dim mismatch: expected {embed_dim}, got {model.embed_dim}"
            
            # Check Depth (number of blocks)
            # CustomVisionTransformer stores blocks in model.blocks
            assert len(model.blocks) == depth, f"Depth mismatch: expected {depth}, got {len(model.blocks)}"
            
            # Check Num Heads (in first block)
            assert model.blocks[0].attn.num_heads == num_heads, f"Num heads mismatch: expected {num_heads}, got {model.blocks[0].attn.num_heads}"
            
            # Check Patch Size
            # Patch embed layer: model.patch_embed.patch_size
            # It returns a tuple (H, W)
            ps = model.patch_embed.patch_size
            assert ps[0] == patch_size and ps[1] == patch_size, f"Patch size mismatch: expected {patch_size}, got {ps}"

            print(f"Verified {name} successfully.")
        except Exception as e:
            print(f"Failed to verify {name}: {e}")
            raise e

    print("\nAll configurations verified!")

if __name__ == "__main__":
    test_configs()
