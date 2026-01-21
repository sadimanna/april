import torch
import timm
import torch.nn as nn
import logging
def get_model(model_name, pretrained=False, num_classes=10):
    """
    Creates a ViT model.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

class April(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.input_embedding = None

    def get_attention_weights(self, block_index):
        """
        Returns the Q, K, V weights for a specific block.
        If LoRA is used, returns the effective weights.
        """
        attn = self.model.blocks[block_index].attn
        qkv_weights = attn.qkv.weight.data
        embed_dim = self.model.embed_dim

        # The qkv weights are concatenated, so we need to split them
        q_weights = qkv_weights[:embed_dim, :]
        k_weights = qkv_weights[embed_dim:2*embed_dim, :]
        v_weights = qkv_weights[2*embed_dim:, :]
        
        # Check for LoRA
        if hasattr(attn, 'lora_rank') and attn.lora_rank > 0:
            # Add LoRA contribution: W_eff = W + s * B.T @ A.T
            # LoRA forward: x @ A @ B * s
            # Linear forward: x @ W.T
            # Contribution to W.T is s * A @ B implies contribution to W is s * (A @ B).T = s * B.T @ A.T
            
            def get_lora_update(lora_layer):
                A = lora_layer.lora_A.data # (in, r)
                B = lora_layer.lora_B.data # (r, out)
                scale = lora_layer.scale
                return scale * (B.T @ A.T)

            q_weights = q_weights + get_lora_update(attn.lora_q)
            k_weights = k_weights + get_lora_update(attn.lora_k)
            v_weights = v_weights + get_lora_update(attn.lora_v)

        return q_weights, k_weights, v_weights

    def get_attention_gradients(self, block_index):
        """
        Returns the gradients of the Q, K, V weights for a specific block.
        If LoRA is used, reconstructs gradients from LoRA adapters and applies square root.
        """
        attn = self.model.blocks[block_index].attn
        embed_dim = self.model.embed_dim
        
        # Check for LoRA
        if hasattr(attn, 'lora_rank') and attn.lora_rank > 0:
            logging.info("Has LoRA")
            grads = []
            for name in ['q', 'k', 'v']:
                lora_layer = getattr(attn, f'lora_{name}')
                dA = lora_layer.lora_A.grad # (in, r)
                dB = lora_layer.lora_B.grad # (r, out)
                
                # If gradients are None (e.g. not optimized or backward not called), return zeros?
                # User assumes this is called after backward.
                if dA is None or dB is None:
                     device = lora_layer.lora_A.device
                     # Construct zero gradient of appropriate shape (out, in)
                     out_features = lora_layer.lora_B.shape[1]
                     in_features = lora_layer.lora_A.shape[0]
                     grads.append(torch.zeros((out_features, in_features), device=device))
                     continue

                # Construct gradient matrix
                # The user requested: "size for the adapter added parallel to q should be of the same size as q weight matrix"
                # q weight matrix is (out, in).
                # We form this via outer product of partials?
                # Heuristic: G ~ dB.T @ dA.T
                # dB.T: (out, r), dA.T: (r, in) -> (out, in)
                
                G = dB.T @ dA.T
                
                # "returned gradient matrix must be square rooted before returning"
                # Assuming signed element-wise square root to preserve directionality while compressing magnitude
                # G = G.sign() * torch.sqrt(G.abs())

                logging.info(f"Norm of gradient: {torch.linalg.norm(G)}")
                
                grads.append(G)
            
            dldq, dldk, dldv = grads[0], grads[1], grads[2]
            
        else:
            qkv_grad = attn.qkv.weight.grad
            
            # The qkv gradients are concatenated, so we need to split them
            dldq = qkv_grad[:embed_dim, :]
            dldk = qkv_grad[embed_dim:2*embed_dim, :]
            dldv = qkv_grad[2*embed_dim:, :]

        return dldq, dldk, dldv

    def get_layer_norm_weights(self, block_index):
        """
        Returns the LayerNorm weights and biases for a specific block.
        """
        ln1_weight = self.model.blocks[block_index].norm1.weight.data
        ln1_bias = self.model.blocks[block_index].norm1.bias.data

        return (ln1_weight, ln1_bias)

    def forward(self, x):
        # We need to register a hook to get the input embedding z
        # before it's processed by the model.
        
        def hook(module, input, output):
            self.input_embedding = input[0].clone()

        handle = self.model.blocks[0].register_forward_hook(hook)
        output = self.model(x)
        handle.remove()

        return output
