import torch
import numpy as np
import config
from dataloaders import get_dataloader
from models.model import get_model, April
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime

def april(w, dldw, dldz):
    """
    Implements the APRIL algorithm for all patches in a batch.
    """
    q_weights = w['q'] #.cpu().numpy()
    k_weights = w['k'] #.cpu().numpy()
    v_weights = w['v'] #.cpu().numpy()

    dldq = dldw['q'] #.cpu().numpy()
    dldk = dldw['k'] #.cpu().numpy()
    dldv = dldw['v'] #.cpu().numpy()

    print("\n--- APRIL Algorithm ---")
    print(f"q_weights range: min={q_weights.min():.4f}, max={q_weights.max():.4f}")
    print(f"k_weights range: min={k_weights.min():.4f}, max={k_weights.max():.4f}")
    print(f"v_weights range: min={v_weights.min():.4f}, max={v_weights.max():.4f}")
    print(f"dldq range: min={dldq.min():.4f}, max={dldq.max():.4f}")
    print(f"dldk range: min={dldk.min():.4f}, max={dldk.max():.4f}")
    print(f"dldv range: min={dldv.min():.4f}, max={dldv.max():.4f}")
    if dldz is not None:
        print(f"dldz range: min={dldz.min().item():.4f}, max={dldz.max().item():.4f}")
    A = q_weights.T @ dldq
    B = k_weights.T @ dldk
    C = v_weights.T @ dldv
    print(f"shape of A: {A.shape}, range: min={A.min():.4f}, max={A.max():.4f}")
    print(f"shape of B: {B.shape}, range: min={B.min():.4f}, max={B.max():.4f}")
    print(f"shape of C: {C.shape}, range: min={C.min():.4f}, max={C.max():.4f}")

    b = A + B + C
    print(f"shape of RHS sum: {b.shape}")
    print(f"RHS sum range: min={b.min():.4f}, max={b.max():.4f}")

    if dldz is None:
        return None

    batch_size, seq_len, embed_dim = dldz.shape
    dldz_reshaped = dldz.view(batch_size * seq_len, embed_dim)
    
    # We are solving Ax=b where A is a row vector, x is a column vector, b is a scalar.
    # x = A_pinv * b

    A_pinv = torch.linalg.pinv(dldz_reshaped)  # (N, D)
    print(f"Shape of pseudo-inverse of A: {A_pinv.shape}")
    print(f"A_pinv range: min={A_pinv.min().item():.4f}, max={A_pinv.max().item():.4f}")

    z_reconstructed = A_pinv.T @ b.to(device=A_pinv.device) # (D, N)
    print(f"Shape of reconstructed z: {z_reconstructed.shape}")
    print(f"z_reconstructed range: min={z_reconstructed.min():.4f}, max={z_reconstructed.max():.4f}")
    z_reconstructed = z_reconstructed.reshape(batch_size, seq_len, embed_dim)

    return z_reconstructed


if __name__ == '__main__':
    # 1. Load the dataset
    dataloader = get_dataloader(config.DEFAULT_DATASET, root='./data')

    # 2. Create the model
    vit_model = get_model(config.DEFAULT_MODEL, 
                          pretrained= True, 
                          num_classes=config.DEFAULT_NUM_CLASSES)
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
        global ln_output_embedding
        ln_input_embedding = input_val[0] # We capture the input to the first layer norm
        ln_output_embedding = output_val.clone() # We capture the output of the first layer norm -- this is z_reconstructed

    handle_forward = model.model.patch_embed.register_forward_hook(hook_forward)
    handle_ln_input = model.model.blocks[0].norm1.register_forward_hook(hook_ln_input)

    outputs = model(images)

    handle_forward.remove()
    handle_ln_input.remove()

    # We need to get the derivative of the loss with respect to the input embedding z.
    # We use torch.autograd.grad to explicitly compute this gradient.
    loss = criterion(outputs, labels)

    # Get the gradient of the loss w.r.t. ln_input_embedding
    # BUT IS THIS POSSIBLE FOR THE SERVER? IN THAT CASE HOOKING BECOMES A CASE OF MODEL POISONING?
    # CAN IT BE OBTAINEDUSING SOME ANALYTICAL METHOD? OR IS IT JUST ASSUMED THAT THE SERVER HAS ACCESS TO IT?
    input_embedding_grad = None
    if ln_input_embedding is not None:
        input_embedding_grad = torch.autograd.grad(loss, ln_input_embedding, retain_graph=True)[0]
        print(f"Shape of input_embedding_grad: {input_embedding_grad.shape}")
        print(f"Range of input_embedding_grad: min={input_embedding_grad.min().item():.4f}, max={input_embedding_grad.max().item():.4f}")

    loss.backward()
    optimizer.step()

    # 5. Get the weights and gradients for a specific block (e.g., block 0)
    block_index = 0
    q_weights, k_weights, v_weights = model.get_attention_weights(block_index) # The server has access to these
    dldq, dldk, dldv = model.get_attention_gradients(block_index) # The server has access to these

    w = {'q': q_weights, 'k': k_weights, 'v': v_weights}
    dldw = {'q': dldq, 'k': dldk, 'v': dldv}

    # 6. Apply the APRIL algorithm
    # print(input_embedding_grad)
    z_reconstructed = april(w, dldw, input_embedding_grad)

    print("\n--- Patch Embedding Reconstruction Results ---")

    # 7. Compare the reconstructed embedding with the original
    original_embedding = ln_output_embedding #.cpu().detach().numpy()
    
    print("Original embedding shape:", original_embedding.shape)
    print("Reconstructed embedding shape:", z_reconstructed.shape)

    # Calculate the error
    error = torch.linalg.norm(original_embedding - z_reconstructed)
    print(f"Reconstruction error: {error}")

    # --- Embedding Visualization ---
    print("\n--- Embedding Visualization ---")
    
    batch_index = 0
    
    # Exclude the CLS token
    original_emb_patches = original_embedding[batch_index, 1:, :]
    recon_emb_patches = z_reconstructed[batch_index, 1:, :]
    
    # Calculate the L2 norm of the embeddings for visualization
    original_emb_norm = torch.linalg.norm(original_emb_patches, axis=-1).detach().cpu().numpy()
    recon_emb_norm = torch.linalg.norm(recon_emb_patches, axis=-1).detach().cpu().numpy()
    print(f"Original embedding norm shape: {original_emb_norm.shape}")
    print(f"Reconstructed embedding norm shape: {recon_emb_norm.shape}")
    # Get the patch grid dimensions
    num_patches = original_emb_patches.shape[0]
    num_patches_per_dim = int(np.sqrt(num_patches))
    
    # Reshape to a 2D grid
    original_emb_grid = original_emb_norm.reshape(num_patches_per_dim, num_patches_per_dim)
    recon_emb_grid = recon_emb_norm.reshape(num_patches_per_dim, num_patches_per_dim)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    im1 = axes[0].imshow(original_emb_grid, cmap='viridis')
    axes[0].set_title("Original Embedding Norm")
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(recon_emb_grid, cmap='viridis')
    axes[1].set_title("Reconstructed Embedding Norm")
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig('embedding_visualization.png')
    print("Saved embedding visualization to embedding_visualization.png")



    # 8. Reconstruct image from intermediate embedding
    print("\n--- Image Patch Reconstruction ---")
    
    # The user requested to invert the LayerNorm using its weights and biases,
    # and the mean and standard deviation from the input images.
    
    # Step 1: Invert the LayerNorm
    norm_layer = model.model.blocks[0].norm1
    gamma = norm_layer.weight
    beta = norm_layer.bias

    # Assuming z_reconstructed is the output of the LayerNorm
    y = z_reconstructed.to(config.DEFAULT_DEVICE)

    # Using mean and std from input images, as requested.
    # Note: LayerNorm's statistics are typically calculated per-token on the embedding dimension.
    # Using global image statistics is not standard and may lead to unexpected results.
    img_mean = images.mean(dim = (1,2,3)).view(-1,1,1)
    img_std = images.std(dim = (1,2,3)).view(-1,1,1)
    
    # Invert the LayerNorm equation: x = ((y - beta) / gamma) * std + mean
    print(f"Range of 'y'(=z_recons..)(b4 inversion): min={y.min().item():.4f}, max={y.max().item():.4f}")
    print(f"Range of original 'y'(=ln_output_emb..): min={ln_output_embedding.min().item():.4f}, max={ln_output_embedding.max().item():.4f}")
    ln_input_embedding_ = ((y - beta) / gamma) * img_std + img_mean
    print(f"Range of 'ln_input_emb..'(after inver.): min={ln_input_embedding_.min().item():.4f}, max={ln_input_embedding_.max().item():.4f}")
    print(f"Range of original 'ln_input_embedding': min={ln_input_embedding.min().item():.4f}, max={ln_input_embedding.max().item():.4f}")
    print(f"Shape of inverted ln_input_embedding: {ln_input_embedding_.shape}")
    ln_input_recon_error = torch.norm(ln_input_embedding - ln_input_embedding_).item()
    print(f"LayerNorm inversion error: {ln_input_recon_error}")

    # Step 2: Subtract the positional embedding to get the patch information.
    # The input to the first transformer block is P' = [x_class; x_p^1; ...; x_p^N] + E_pos
    # So, P' - E_pos = [x_class; x_p^1; ...; x_p^N]
    # ln_input_embedding_ = z_reconstructed.to(config.DEFAULT_DEVICE)
    pos_embed = model.model.pos_embed
    print(f"Shape of positional embedding: {pos_embed.shape}")
    print(f"Range of positional embedding: min={pos_embed.min().item():.4f}, max={pos_embed.max().item():.4f}")
    z_patch_with_cls = ln_input_embedding_ - pos_embed
    
    # Remove the class token
    z_patch = z_patch_with_cls[:, 1:, :]
    print(f"Shape of patch embeddings (z_patch): {z_patch.shape}")
    print(f"Range of patch embeddings (z_patch): min={z_patch.min().item():.4f}, max={z_patch.max().item():.4f}")

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
    nowtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'reconstructed_images_{nowtime}.png')

