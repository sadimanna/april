import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
from timm.layers import PatchEmbed
from .vit_config import VIT_CONFIGS

class CustomBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=False, proj_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layernorm=True, use_residual=True):
        super(CustomBlock, self).__init__(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                                           proj_drop=proj_drop, attn_drop=attn_drop, drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer)
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual

        if not self.use_layernorm:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x):
        if self.use_residual:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        else:
            x = self.drop_path1(self.ls1(self.attn(self.norm1(x))))
            x = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CustomVisionTransformer(TimmVisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=None,
                 norm_layer=None, act_layer=None, weight_init='', use_layernorm=True, use_residual=True, global_pool=''):
        
        # Default act_layer to nn.GELU if None
        act_layer = act_layer or nn.GELU

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
                use_residual=use_residual if i == 0 else True)
            for i in range(depth)])

def get_custom_vit(model_name='vit_base_patch16_224', pretrained=False, **kwargs):
    config = VIT_CONFIGS.get(model_name)
    if config is None:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(VIT_CONFIGS.keys())}")

    model = CustomVisionTransformer(
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        embed_layer=PatchEmbed,
        global_pool='token',
        **kwargs
    )

    if pretrained:
        from timm.models import create_model
        timm_model = create_model(model_name, pretrained=True)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in timm_model.state_dict().items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

if __name__ == '__main__':
    # Example of how to use the custom ViT model
    
    dummy_input = torch.randn(2, 3, 224, 224)

    # --- ViT Base ---
    print("--- ViT Base ---")
    vit_base_standard = get_custom_vit(model_name='vit_base_patch16_224', use_layernorm=True, use_residual=True)
    output_standard = vit_base_standard(dummy_input)
    print(output_standard.shape)

    # --- ViT Tiny ---
    print("\n--- ViT Tiny ---")
    vit_tiny_standard = get_custom_vit(model_name='vit_tiny_patch16_224', use_layernorm=True, use_residual=True)
    output_tiny = vit_tiny_standard(dummy_input)
    print(output_tiny.shape)

    # --- ViT Base without LayerNorm (First Layer Only) ---
    print("\n--- ViT Base without LayerNorm (First Layer Only) ---")
    vit_base_no_ln = get_custom_vit(model_name='vit_base_patch16_224', use_layernorm=False, use_residual=True)
    output_no_ln = vit_base_no_ln(dummy_input)
    print(output_no_ln.shape)

    # --- ViT Base without Residual Connection (First Layer Only) ---
    print("\n--- ViT Base without Residual Connection (First Layer Only) ---")
    vit_base_no_res = get_custom_vit(model_name='vit_base_patch16_224', use_layernorm=True, use_residual=False)
    output_no_res = vit_base_no_res(dummy_input)
    print(output_no_res.shape)

    # --- ViT Base without LayerNorm and Residual Connection (First Layer Only) ---
    print("\n--- ViT Base without LayerNorm and Residual Connection (First Layer Only) ---")
    vit_base_minimal = get_custom_vit(model_name='vit_base_patch16_224', use_layernorm=False, use_residual=False)
    output_minimal = vit_base_minimal(dummy_input)
    print(output_minimal.shape)
