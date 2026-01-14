import torch
import numpy as np
import config
from dataloaders import get_dataloader
from models.model import get_model, April
import torch.nn as nn

def april(w, dldw, dldz):
    """
    Implements the APRIL algorithm.
    This implementation is based on the description in the prompt, but with some assumptions
    due to the ambiguity of the mathematical operations.
    """
    q_weights = w['q'].cpu().numpy()
    k_weights = w['k'].cpu().numpy()
    v_weights = w['v'].cpu().numpy()

    dldq = dldw['q'].cpu().numpy()
    dldk = dldw['k'].cpu().numpy()
    dldv = dldw['v'].cpu().numpy()

    # The prompt states: b = sum(W.T * dLdW)
    # The shapes of W.T and dLdW are not compatible for matrix multiplication.
    # Let's assume it's an element-wise product followed by a sum over all elements.
    # This would result in a scalar value for b.
    b = np.sum(q_weights.T * dldq) + np.sum(k_weights.T * dldk) + np.sum(v_weights.T * dldv)

    # The prompt states: Az = b, where A = dLdz.
    # If b is a scalar, and z is a vector of shape (d_model, 1), then A must be a row vector of shape (1, d_model).
    # dldz has shape (batch_size, seq_len, embed_dim).
    # We will take the first element of the batch and sequence.
    A = dldz[0, 0, :].cpu().numpy()

    # Now we have A (1, d_model) and z (d_model, 1), so Az is a scalar.
    # We can solve for z using the pseudoinverse of A.
    A_pinv = np.linalg.pinv(A.reshape(1, -1)) # Reshape to ensure it's a row vector
    
    # b is a scalar, so we need to make it a 1x1 matrix for the dot product.
    b_matrix = np.array([[b]])
    
    z_rec = np.dot(A_pinv, b_matrix)

    return z_rec.T


if __name__ == '__main__':
    # 1. Load the dataset
    dataloader = get_dataloader(config.DEFAULT_DATASET, root='./data')

    # 2. Create the model
    vit_model = get_model(config.DEFAULT_MODEL, num_classes=10)
    model = April(vit_model)
    model.to(config.DEFAULT_DEVICE)

    # 3. Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.DEFAULT_LR)

    # Get a single batch of data
    images, labels = next(iter(dataloader))
    images = images.to(config.DEFAULT_DEVICE)
    labels = labels.to(config.DEFAULT_DEVICE)

    # 4. Train the model for one step
    optimizer.zero_grad()
    
    # Register a hook to get the input embedding z
    input_embedding = None
    def hook_forward(module, input, output):
        global input_embedding
        input_embedding = input[0].clone()
    
    handle_forward = model.model.patch_embed.register_forward_hook(hook_forward)
    
    outputs = model(images)
    
    handle_forward.remove()

    # We need to get the derivative of the loss with respect to the input embedding z.
    # We can do this by registering a backward hook on the patch embedding layer.
    
    input_embedding_grad = None
    def hook_grad(module, grad_input, grad_output):
        global input_embedding_grad
        input_embedding_grad = grad_input[0]

    handle_backward = model.model.patch_embed.register_full_backward_hook(hook_grad)
    
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    handle_backward.remove()

    # 5. Get the weights and gradients for a specific block (e.g., block 0)
    block_index = 0
    q_weights, k_weights, v_weights = model.get_attention_weights(block_index)
    dldq, dldk, dldv = model.get_attention_gradients(block_index)

    w = {'q': q_weights, 'k': k_weights, 'v': v_weights}
    dldw = {'q': dldq, 'k': dldk, 'v': dldv}

    # 6. Apply the APRIL algorithm
    z_reconstructed = april(w, dldw, input_embedding_grad)

    # 7. Compare the reconstructed embedding with the original
    original_embedding = input_embedding.cpu().detach().numpy()
    
    original_z = original_embedding[0, 0, :]
    
    print("Original z shape:", original_z.shape)
    print("Reconstructed z shape:", z_reconstructed.shape)
    
    # Calculate the error
    error = np.linalg.norm(original_z - z_reconstructed)
    print(f"Reconstruction error: {error}")
