import torch
import torch.nn as nn
from timm.models.vision_transformer import Block as TimmBlock
from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer

class CustomBlock(TimmBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_layernorm=True, use_residual=True):
        super(CustomBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                           drop_path, act_layer, norm_layer)
        self.use_layernorm = use_layernorm
        self.use_residual = use_residual

        if not self.use_layernorm:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

    def forward(self, x):
        if self.use_residual:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = self.drop_path(self.attn(self.norm1(x)))
            x = self.drop_path(self.mlp(self.norm2(x)))
        return x

class CustomVisionTransformer(TimmVisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 distilled=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=None,
                 norm_layer=None, act_layer=None, weight_init='', use_layernorm=True, use_residual=True):
        super(CustomVisionTransformer, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dim, depth,
                                                     num_heads, mlp_ratio, qkv_bias, qk_scale, representation_size,
                                                     distilled, drop_rate, attn_drop_rate, drop_path_rate, embed_layer,
                                                     norm_layer, act_layer, weight_init)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            CustomBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                use_layernorm=use_layernorm, use_residual=use_residual)
            for i in range(depth)])

def get_custom_vit(model_name='vit_base_patch16_224', pretrained=False, **kwargs):
    model = CustomVisionTransformer(
                                    patch_size=16, 
                                    embed_dim=768, 
                                    depth=12, 
                                    num_heads=12, 
                                    mlp_ratio=4, 
                                    qkv_bias=True,
                                    norm_layer=nn.LayerNorm, 
                                    **kwargs)
    
    if pretrained:
        # Load pretrained weights from timm
        from timm.models import create_model
        timm_model = create_model(model_name, pretrained=True)
        model.load_state_dict(timm_model.state_dict(), strict=False)
        
    return model

if __name__ == '__main__':
    # Example of how to use the custom ViT model
    
    # Create a model with LayerNorm and residual connections (standard ViT)
    vit_standard = get_custom_vit(use_layernorm=True, use_residual=True)
    
    # Create a model without LayerNorm
    vit_no_layernorm = get_custom_vit(use_layernorm=False, use_residual=True)
    
    # Create a model without residual connections
    vit_no_residual = get_custom_vit(use_layernorm=True, use_residual=False)
    
    # Create a model without both
    vit_minimal = get_custom_vit(use_layernorm=False, use_residual=False)
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("--- Standard ViT ---")
    output_standard = vit_standard(dummy_input)
    print(output_standard.shape)
    
    print("\n--- ViT without LayerNorm ---")
    output_no_ln = vit_no_layernorm(dummy_input)
    print(output_no_ln.shape)
    
    print("\n--- ViT without Residual Connection ---")
    output_no_res = vit_no_residual(dummy_input)
    print(output_no_res.shape)

    print("\n--- ViT without LayerNorm and Residual Connection ---")
    output_minimal = vit_minimal(dummy_input)
    print(output_minimal.shape)
