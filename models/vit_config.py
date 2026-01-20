VIT_CONFIGS = {
    'vit_tiny_patch16_224': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'patch_size': 16,
    },
    'vit_base_patch16_224': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'patch_size': 16,
    },
    'vit_small_cifar': {
        'embed_dim': 384,
        'depth': 4,
        'num_heads': 4,
        'patch_size': 16, # 32x32 image / 16x16 patch = 2x2 grid = 4 patches
    },
    'vit_small_mnist': {
        'embed_dim': 384,
        'depth': 4,
        'num_heads': 4,
        'patch_size': 14, # 28x28 image / 14x14 patch = 2x2 grid = 4 patches
    },
    'vit_medium_stl10': {
        'embed_dim': 384,
        'depth': 6,
        'num_heads': 6,
        'patch_size': 16,
    },
}
