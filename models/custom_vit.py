import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer, checkpoint_seq
from timm.layers import PatchEmbed
from .vit_config import VIT_CONFIGS
from functools import partial
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(in_features, rank), requires_grad = True)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features), requires_grad = True)
        # Perturb lora_B to avoid zero gradients (singularity at init)
        # with torch.no_grad():

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        # nn.init.kaiming_uniform_(self.lora_B, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        # self.lora_B.data = self.lora_B.data + torch.randn_like(self.lora_B.data) * 0.01

    def forward(self, x):
        # x: (batch_size, seq_len, in_features)
        # lora_A: (in_features, rank)
        # lora_B: (rank, out_features)
        return (x @ self.lora_A @ self.lora_B) * self.scale

class CustomPatchEmbed(PatchEmbed):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, patch_embed_type='conv', **kwargs):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, norm_layer=norm_layer, flatten=flatten, **kwargs)
        self.patch_embed_type = patch_embed_type
        
        if patch_embed_type == 'linear':
            self.proj = nn.Linear(self.patch_size[0] * self.patch_size[1] * in_chans, embed_dim)
    
    def forward(self, x):
        if self.patch_embed_type == 'linear':
            if x.dim() == 4:
                x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size) # (B, C*P*P, N)
                x = x.transpose(1, 2) # (B, N, C*P*P)
            elif x.dim() == 5:
                # Assume (B, N, C, Ph, Pw)
                B, N, C, Ph, Pw = x.shape
                x = x.reshape(B, N, -1)
            elif x.dim() == 3:
                # Assume (B, N, D)
                pass
            else:
                raise ValueError(f"Unsupported input shape for Linear PatchEmbed: {x.shape}. Expected 3D, 4D, or 5D.")
            
            x = self.proj(x)
            if self.norm is not None:
                x = self.norm(x)
            return x
        else:
            if x.dim() != 4:
                 raise ValueError(f"Conv PatchEmbed only supports 4D input (B, C, H, W). Got {x.dim()}D: {x.shape}")
            return super().forward(x)


class CustomAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm, lora_rank=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_q = LoRALayer(dim, dim, rank=lora_rank, alpha = 8)
            self.lora_k = LoRALayer(dim, dim, rank=lora_rank, alpha = 8)
            self.lora_v = LoRALayer(dim, dim, rank=lora_rank, alpha = 8)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        if self.lora_rank > 0:
            # Apply LoRA to q, k, v
            # q, k, v are (B, num_heads, N, head_dim)
            # Need to reshape/permute to apply LoRA which expects (B, N, C)
            # LoRA forward returns (B, N, C)
            q_lora = self.lora_q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            k_lora = self.lora_k(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            v_lora = self.lora_v(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            q = q + q_lora
            k = k + k_lora
            v = v + v_lora

        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        if attn_mask is not None:
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CustomBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layernorm=True, use_residual=True, lora_rank=0):
        super(CustomBlock, self).__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                                           proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual
        
        # Override attention with CustomAttention
        self.attn = CustomAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm,
            attn_drop=attn_drop, proj_drop=proj_drop, norm_layer=norm_layer, lora_rank=lora_rank
        )

        if not self.use_layernorm:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x, attn_mask=None):
        if self.use_residual:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask)))
            x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CustomVisionTransformer(TimmVisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=None,
                 norm_layer=None, act_layer=None, weight_init='', use_layernorm=True, use_residual=True, 
                 global_pool='', lora_rank=0, patch_embed_type='conv'):
        
        # Default act_layer to nn.GELU if None
        act_layer = act_layer or nn.GELU

        # Use partial to pass patch_embed_type to CustomPatchEmbed
        embed_layer = partial(CustomPatchEmbed, img_size=img_size, patch_embed_type=patch_embed_type) if embed_layer is None or embed_layer == PatchEmbed or embed_layer == CustomPatchEmbed else embed_layer

        super(CustomVisionTransformer, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                                                     num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                     drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                                                     norm_layer=norm_layer, act_layer=act_layer, weight_init=weight_init, global_pool=global_pool)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        
        # Re-create blocks using CustomBlock
        # Apply use_layernorm and use_residual ONLY to the first block (index 0)
        # For all other blocks, default to True (standard ViT behavior)
        self.blocks = nn.Sequential(*[
            CustomBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=False,
                proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                use_layernorm=use_layernorm if i == 0 else True,
                use_residual=use_residual if i == 0 else True,
                lora_rank=lora_rank)
            for i in range(depth)])

    def _process_input(self, x):
        """
        Check and reshape input to align with the PatchEmbed model requirements.
        """
        if self.patch_embed.patch_embed_type == 'conv':
            if x.dim() != 4:
                # Attempt to reshape 3D/5D to 4D if possible, or raise error
                if x.dim() == 5:
                     # (B, N, C, Ph, Pw) -> (B, C, H, W)
                     # This is complex without knowing grid size, so we might just warn/error
                     pass
        return x

    def forward_features(self, x, attn_mask=None):
        x = self._process_input(x)
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        
        if attn_mask is not None:
            # If mask provided, we need to apply blocks one by one
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x

def get_custom_vit(model_name='vit_base_patch16_224', pretrained=False, lora_rank=0, patch_embed_type='conv', **kwargs):
    config = VIT_CONFIGS.get(model_name)
    if config is None:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(VIT_CONFIGS.keys())}")

    model = CustomVisionTransformer(
        img_size=config.get('img_size', 224),
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        embed_layer=CustomPatchEmbed,
        global_pool='token',
        lora_rank=lora_rank,
        patch_embed_type=patch_embed_type,
        in_chans = 1 if 'mnist' in model_name else 3,
        **kwargs
    )

    # if pretrained:
    #     from timm.models import create_model
    #     timm_model = create_model(model_name, pretrained=True)
    #     model_dict = model.state_dict()
    #     pretrained_dict = {k: v for k, v in timm_model.state_dict().items() if k in model_dict and v.shape == model_dict[k].shape}
    #     model_dict.update(pretrained_dict)
    #     model.load_state_dict(model_dict, strict=False)

    return model

if __name__ == '__main__':
    # Example of how to use the custom ViT model
    
    dummy_input = torch.randn(2, 3, 224, 224)

    # --- ViT Base ---
    print("--- ViT Base with LoRA ---")
    vit_base_lora = get_custom_vit(model_name='vit_base_patch16_224', use_layernorm=True, use_residual=True, lora_rank=4)
    output_lora = vit_base_lora(dummy_input)
    print(output_lora.shape)
    
    # Check if LoRA parameters are present
    print("\nLoRA Parameters:")
    for name, param in vit_base_lora.named_parameters():
        if 'lora' in name:
            print(name, param.shape)
            
    # --- ViT Base with Linear PatchEmbed ---
    print("\n--- ViT Base with Linear PatchEmbed ---")
    vit_linear = get_custom_vit(model_name='vit_base_patch16_224', patch_embed_type='linear', lora_rank=0)
    output_linear = vit_linear(dummy_input)
    print("Output shape:", output_linear.shape)
    print("Patch embed type:", vit_linear.patch_embed.patch_embed_type)
    print("Patch embed proj:", vit_linear.patch_embed.proj)

    # --- Test with 3D Input (Flattened Patches) ---
    print("\n--- ViT Base with Linear PatchEmbed (3D Input) ---")
    # Patch size 16, 3 channels -> 768 features. 224/16 = 14 -> 14x14 = 196 patches.
    dummy_input_3d = torch.randn(2, 196, 768)
    output_3d = vit_linear(dummy_input_3d)
    print("3D Input Output shape:", output_3d.shape)

    # --- Test with 5D Input (Batched Patches) ---
    print("\n--- ViT Base with Linear PatchEmbed (5D Input) ---")
    dummy_input_5d = torch.randn(2, 196, 3, 16, 16)
    output_5d = vit_linear(dummy_input_5d)
    print("5D Input Output shape:", output_5d.shape)


