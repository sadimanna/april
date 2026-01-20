import torch
from models.custom_vit import get_custom_vit

model = get_custom_vit()
print(model.blocks[0].attn)
for name, module in model.blocks[0].attn.named_modules():
    print(name, module)
