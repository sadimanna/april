import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.randn(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scale

# Input
x = torch.randn(2, 5, 10) # (B, N, in_features)
lora = LoRALayer(10, 20, rank=4)

# Target
target = torch.randn(2, 5, 20)

# Forward pass
output = lora(x)
loss = F.mse_loss(output, target)

# Backward pass
loss.backward()

print(f"lora_B initialization (first few): {lora.lora_B.flatten()[:5]}")
print(f"lora_A grad norm: {lora.lora_A.grad.norm().item()}")
print(f"lora_B grad norm: {lora.lora_B.grad.norm().item()}")

# Optimizer step
optimizer = torch.optim.Adam(lora.parameters(), lr=1e-3)
optimizer.step()

# Second forward pass
optimizer.zero_grad()
output = lora(x)
loss = F.mse_loss(output, target)
loss.backward()

print("\n--- After one optimizer step ---")
print(f"lora_B norm: {lora.lora_B.norm().item()}")
print(f"lora_A grad norm: {lora.lora_A.grad.norm().item()}")
print(f"lora_B grad norm: {lora.lora_B.grad.norm().item()}")
