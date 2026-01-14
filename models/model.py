import torch
import timm
import torch.nn as nn

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
        """
        qkv_weights = self.model.blocks[block_index].attn.qkv.weight.data
        embed_dim = self.model.embed_dim

        # The qkv weights are concatenated, so we need to split them
        q_weights = qkv_weights[:embed_dim, :]
        k_weights = qkv_weights[embed_dim:2*embed_dim, :]
        v_weights = qkv_weights[2*embed_dim:, :]

        return q_weights, k_weights, v_weights


    def get_attention_gradients(self, block_index):
        """
        Returns the gradients of the Q, K, V weights for a specific block.
        """
        qkv_grad = self.model.blocks[block_index].attn.qkv.weight.grad
        embed_dim = self.model.embed_dim

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
