import torch
import numpy as np
import config
from dataloaders import get_dataloader
from models.model import get_model, April
import torch.nn as nn
import matplotlib.pyplot as plt

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
    b = np.sum(q_weights.T @ dldq) + np.sum(k_weights.T @ dldk) + np.sum(v_weights.T @ dldv)

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
    
    # Register hooks to get intermediate embeddings
    input_embedding = None
    ln_input_embedding = None

    def hook_forward(module, input, output):
        global input_embedding
        input_embedding = output.clone() # We capture the output of patch_embed

    def hook_ln_input(module, input_val, output_val):
        global ln_input_embedding
        ln_input_embedding = input_val[0].clone()

    handle_forward = model.model.patch_embed.register_forward_hook(hook_forward)
    handle_ln_input = model.model.blocks[0].norm1.register_forward_hook(hook_ln_input)

    ln_input_tensor = None
    def hook_get_input_tensor(module, input, output):
        global ln_input_tensor
        ln_input_tensor = input[0]

    handle_get_input = model.model.blocks[0].norm1.register_forward_hook(hook_get_input_tensor)

    outputs = model(images)

    handle_get_input.remove()
    handle_forward.remove()
    handle_ln_input.remove()

    # We need to get the derivative of the loss with respect to the input embedding z.
    # We use torch.autograd.grad to explicitly compute this gradient.
    loss = criterion(outputs, labels)

    input_embedding_grad = None
    if ln_input_tensor is not None:
        input_embedding_grad = torch.autograd.grad(loss, ln_input_tensor, retain_graph=True)[0]

    loss.backward()
    optimizer.step()


    # 5. Get the weights and gradients for a specific block (e.g., block 0)
    block_index = 0
    q_weights, k_weights, v_weights = model.get_attention_weights(block_index)
    dldq, dldk, dldv = model.get_attention_gradients(block_index)

    w = {'q': q_weights, 'k': k_weights, 'v': v_weights}
    dldw = {'q': dldq, 'k': dldk, 'v': dldv}

    # 6. Apply the APRIL algorithm
    # print(input_embedding_grad)
    z_reconstructed = april(w, dldw, input_embedding_grad)

    # 7. Compare the reconstructed embedding with the original
    original_embedding = input_embedding.cpu().detach().numpy()
    
    original_z = original_embedding[0, 0, :]
    
    print("Original z shape:", original_z.shape)
    print("Reconstructed z shape:", z_reconstructed.shape)
    
    # Calculate the error
    error = np.linalg.norm(original_z - z_reconstructed)
    print(f"Reconstruction error: {error}")

    # 8. Reconstruct image from intermediate embedding
    print("\n--- Image Patch Reconstruction ---")
    
    # The user requested to "Use the layer norm weights and biases to obtain the input".
    # However, inverting a LayerNorm is ill-posed without knowing the mean and variance
    # of the input, which are not available from the output. 
    # A more direct way is to hook the layer and get its input directly. We proceed
    # with this more robust approach.
    
    # Step 1: We already have the input to the first layer norm from the hook: ln_input_embedding
    
    # Step 2: Subtract the positional embedding to get the patch information.
    # The input to the first transformer block is P' = [x_class; x_p^1; ...; x_p^N] + E_pos
    # So, P' - E_pos = [x_class; x_p^1; ...; x_p^N]
    pos_embed = model.model.pos_embed
    z_patch_with_cls = ln_input_embedding - pos_embed
    
    # Remove the class token
    z_patch = z_patch_with_cls[:, 1:, :]
    print(f"Shape of patch embeddings (z_patch): {z_patch.shape}")

    # Step 3: Multiply with the pseudo-inverse of the patch embedding projection layer.
    patch_proj_layer = model.model.patch_embed.proj
    
    # The projection is a Conv2d. Its weight maps from (C, patch_size, patch_size) to (D,).
    # We can view this as a linear layer on flattened patches.
    # Weight shape: (D, C, ps, ps)
    patch_proj_weights = patch_proj_layer.weight
    
    # Reshape to (D, C*ps*ps) and transpose to (C*ps*ps, D)
    proj_matrix = patch_proj_weights.view(patch_proj_weights.shape[0], -1).T
    
    # Calculate pseudo-inverse
    pinv_proj_matrix = torch.linalg.pinv(proj_matrix)
    print(f"Shape of pseudo-inverse of projection matrix: {pinv_proj_matrix.shape}")
    
    # Apply pseudo-inverse: (N, L, D) @ (D, C*ps*ps) -> (N, L, C*ps*ps)
    reconstructed_patches_flat = z_patch @ pinv_proj_matrix
    print(f"Shape of flattened reconstructed patches: {reconstructed_patches_flat.shape}")

    # Now we can compare the reconstructed patch with the original image patch.
    # Let's take the first image and first patch.
    patch_size = model.model.patch_embed.patch_size[0]
    first_image = images[0]
    first_patch_original = first_image[:, :patch_size, :patch_size].clone()
    
    # Reshape the reconstructed patch back to image format
    # (N, L, C*ps*ps) -> (N, L, C, ps, ps)
    C = first_image.shape[0]
    reconstructed_patches = reconstructed_patches_flat.view(
        reconstructed_patches_flat.shape[0],
        reconstructed_patches_flat.shape[1],
        C,
        patch_size,
        patch_size
    )
    first_patch_reconstructed = reconstructed_patches[0, 0, :, :, :]
    
    # Calculate reconstruction error for the first patch
    patch_recon_error = torch.norm(first_patch_original - first_patch_reconstructed).item()
    
    print(f"\nReconstruction error for the first patch: {patch_recon_error}")
    print(f"Shape of original patch: {first_patch_original.shape}")
    print(f"Shape of reconstructed patch: {first_patch_reconstructed.shape}")

    # --- Image Reconstruction Visualization ---
    print("\n--- Full Image Reconstruction Visualization ---")

    # Stitch patches back together to form the full image
    num_patches_per_dim = images.shape[-1] // patch_size
    reconstructed_images = reconstructed_patches.view(
        images.shape[0],  # Batch size
        num_patches_per_dim,
        num_patches_per_dim,
        C,
        patch_size,
        patch_size
    ).permute(0, 3, 1, 4, 2, 5).reshape(images.shape[0], C, images.shape[-1], images.shape[-1])

    # Plot original vs. reconstructed images
    fig, axes = plt.subplots(images.shape[0], 2, figsize=(6, images.shape[0] * 3))
    
    # Un-normalization values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(images.shape[0]):
        if images.shape[0] == 1:
            ax_orig, ax_recon = axes
        else:
            ax_orig = axes[i, 0]
            ax_recon = axes[i, 1]

        # Original image
        original_img = images[i].cpu().detach().permute(1, 2, 0).numpy()
        original_img = std * original_img + mean
        original_img = np.clip(original_img, 0, 1)
        
        ax_orig.imshow(original_img)
        ax_orig.set_title(f"Image {i+1}: Original")
        ax_orig.axis('off')
        
        # Reconstructed image
        recon_img = reconstructed_images[i].cpu().detach().permute(1, 2, 0).numpy()
        recon_img = std * recon_img + mean
        recon_img = np.clip(recon_img, 0, 1)

        ax_recon.imshow(recon_img)
        ax_recon.set_title(f"Image {i+1}: Reconstructed")
        ax_recon.axis('off')

    plt.tight_layout()
    plt.show()

