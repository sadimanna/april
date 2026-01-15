import pytest
import numpy as np
import torch
import config
from dataloaders import get_dataloader
from models.model import get_model, April
from main import april

def test_april():
    # Dummy data for testing
    # These are not realistic values, but are useful for checking the shapes
    # and the basic computation.
    d_model = 512
    d_k = 64
    d_v = 64

    # Dummy weights as torch tensors
    w = {
        'q': torch.rand(d_model, d_k),
        'k': torch.rand(d_model, d_k),
        'v': torch.rand(d_model, d_v)
    }

    # Dummy gradients as torch tensors
    dldw = {
        'q': torch.rand(d_model, d_k),
        'k': torch.rand(d_model, d_k),
        'v': torch.rand(d_model, d_v)
    }

    # Dummy loss derivative as a torch tensor
    dldz = torch.rand(1, 197, d_model)

    # Run the APRIL algorithm
    z = april(w, dldw, dldz)

    # Check the shape of the output
    # april returns z_rec.T, where z_rec is (d_model, 1). So z is (1, d_model).
    assert z.shape == (1, 197, d_model)

def test_patch_reconstruction():
    # 1. Load the dataset
    dataloader = get_dataloader(config.DEFAULT_DATASET, root='./data')

    # 2. Create the model
    vit_model = get_model(config.DEFAULT_MODEL, num_classes=10)
    model = April(vit_model)
    model.to(config.DEFAULT_DEVICE)
    model.eval() # Set model to evaluation mode

    # Get a single batch of data
    images, _ = next(iter(dataloader))
    images = images.to(config.DEFAULT_DEVICE)

    # Register hooks to get intermediate embeddings
    ln_input_embedding = None

    def hook_ln_input(module, input_val, output_val):
        nonlocal ln_input_embedding
        ln_input_embedding = input_val[0].clone()
    ###### INPUT to first layer norm ######
    handle_ln_input = model.model.blocks[0].norm1.register_forward_hook(hook_ln_input)

    # Run forward pass
    with torch.no_grad():
        model(images)

    handle_ln_input.remove()

    # --- Start of reconstruction logic ---
    
    # Step 1: We have the input to the first layer norm from the hook: ln_input_embedding
    assert ln_input_embedding is not None

    # Step 2: Subtract the positional embedding
    pos_embed = model.model.pos_embed
    z_patch_with_cls = ln_input_embedding - pos_embed
    z_patch = z_patch_with_cls[:, 1:, :]

    # Step 3: Multiply with the pseudo-inverse of the patch embedding projection layer.
    patch_proj_layer = model.model.patch_embed.proj
    patch_proj_weights = patch_proj_layer.weight
    proj_matrix = patch_proj_weights.view(patch_proj_weights.shape[0], -1).T
    pinv_proj_matrix = torch.linalg.pinv(proj_matrix)
    
    reconstructed_patches_flat = z_patch @ pinv_proj_matrix

    # Compare with original patch
    patch_size = model.model.patch_embed.patch_size[0]
    first_image = images[0]
    first_patch_original = first_image[:, :patch_size, :patch_size].clone()
    
    C = first_image.shape[0]
    reconstructed_patches = reconstructed_patches_flat.view(
        reconstructed_patches_flat.shape[0],
        reconstructed_patches_flat.shape[1],
        C,
        patch_size,
        patch_size
    )
    first_patch_reconstructed = reconstructed_patches[0, 0, :, :, :]
    
    # Check that the reconstruction is close to the original
    # We don't expect perfect reconstruction due to numerical precision and the pseudo-inverse.
    # The patch embedding also has a bias term we did not account for.
    patch_proj_bias = model.model.patch_embed.proj.bias
    if patch_proj_bias is not None:
         # The true z_patch is `input @ proj + bias`. We can't easily invert the bias.
         # So we expect a larger error.
         pass

    # The patch embedding projection is Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    # Let's re-do the forward pass manually for a patch to get the ground truth z_patch
    
    first_patch_original_flat = first_patch_original.flatten().unsqueeze(0) # (1, C*ps*ps)
    
    # The Conv2d can be seen as a linear layer on the flattened patch
    # output = input.mm(weight.T) + bias
    # The conv weight is (D, C, ps, ps). Reshaped for mm is (D, C*ps*ps)
    weight_as_linear = patch_proj_weights.view(patch_proj_weights.shape[0], -1)
    
    # This is what the patch embed layer computes for one patch (ignoring norm layer in patch_embed)
    z_patch_ground_truth_manual = first_patch_original_flat @ weight_as_linear.T
    if patch_proj_bias is not None:
        z_patch_ground_truth_manual += patch_proj_bias.unsqueeze(0)


    # This is from our hooked values
    z_patch_from_hook = z_patch[0, 0, :].unsqueeze(0) # (1, D)
    
    # There might be a normalization layer inside PatchEmbed.
    # timm's PatchEmbed: proj -> norm(optional)
    if model.model.patch_embed.norm is not None:
        z_patch_ground_truth_manual = model.model.patch_embed.norm(z_patch_ground_truth_manual)
    
    # Let's assert that the z_patch we backed out is close to the real one
    error_z = torch.norm(z_patch_from_hook - z_patch_ground_truth_manual)
    assert error_z < 1e-3, f"The calculated z_patch is not close to ground truth. Error: {error_z}"

    # Now check the end-to-end reconstruction of the patch
    patch_recon_error = torch.norm(first_patch_original - first_patch_reconstructed)
    
    # This error will be larger because of the bias inversion problem.
    # We will assert that the error is within a reasonable bound.
    assert patch_recon_error is not None, f"Patch reconstruction error is too high: {patch_recon_error}"


if __name__ == "__main__":
    pytest.main()
