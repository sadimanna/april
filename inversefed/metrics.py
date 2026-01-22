"""This is code based on https://sudomake.ai/inception-score-explained/."""
import torch
import torchvision

from collections import defaultdict

class InceptionScore(torch.nn.Module):
    """Class that manages and returns the inception score of images."""

    def __init__(self, batch_size=32, setup=dict(device=torch.device('cpu'), dtype=torch.float)):
        """Initialize with setup and target inception batch size."""
        super().__init__()
        self.preprocessing = torch.nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)
        self.model = torchvision.models.inception_v3(pretrained=True).to(**setup)
        self.model.eval()
        self.batch_size = batch_size

    def forward(self, image_batch):
        """Image batch should have dimensions BCHW and should be normalized.

        B should be divisible by self.batch_size.
        """
        B, C, H, W = image_batch.shape
        batches = B // self.batch_size
        scores = []
        for batch in range(batches):
            input = self.preprocessing(image_batch[batch * self.batch_size: (batch + 1) * self.batch_size])
            scores.append(self.model(input))
        prob_yx = torch.nn.functional.softmax(torch.cat(scores, 0), dim=1)
        entropy = torch.where(prob_yx > 0, -prob_yx * prob_yx.log(), torch.zeros_like(prob_yx))
        return entropy.sum()


def psnr(img_batch, ref_batch, batched=False, factor=1.0):
    """Standard PSNR."""
    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref)**2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor**2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

def total_variation_patches(x, P=16):
    """Patch-based anisotropic TV computed on patch boundaries.

    For a patch size `P`, compute the L2 norm of the difference across
    vertical patch boundaries (between rows P*k-1 and P*k) and horizontal
    patch boundaries (between cols P*k-1 and P*k). For each boundary the
    difference is flattened across channels and the remaining spatial
    dimension, producing a per-sample L2 norm. The function returns the
    mean (over the batch) of the summed per-sample norms across all
    boundaries.

    Args:
        x (torch.Tensor): input tensor with shape (B, C, H, W).
        P (int): patch size (default 16).

    Returns:
        torch.Tensor: scalar tensor containing the mean patch-TV penalty.
    """
    B, C, H, W = x.shape

    # Prepare per-sample accumulator
    per_sample = x.new_zeros((B,))

    # Vertical boundaries (along height): compare rows P*k-1 and P*k
    n_vert = H // P
    for k in range(1, n_vert):
        a = x[:, :, P * k, :].reshape(B, -1)       # shape (B, C*W)
        b = x[:, :, P * k - 1, :].reshape(B, -1)   # shape (B, C*W)
        diff = a - b
        norms = torch.norm(diff, p=2, dim=1)      # per-sample L2
        per_sample = per_sample + norms

    # Horizontal boundaries (along width): compare cols P*k-1 and P*k
    n_horiz = W // P
    for k in range(1, n_horiz):
        a = x[:, :, :, P * k].reshape(B, -1)      # shape (B, C*H)
        b = x[:, :, :, P * k - 1].reshape(B, -1)  # shape (B, C*H)
        diff = a - b
        norms = torch.norm(diff, p=2, dim=1)      # per-sample L2
        per_sample = per_sample + norms

    # If there were no boundaries (P larger than H or W), return zero
    if (n_vert <= 1) and (n_horiz <= 1):
        return x.new_tensor(0.0)

    # Return mean over batch of summed per-sample boundary norms
    return per_sample.mean()


def BNLoss(x, target_mean=0.0, target_var=1.0, eps=1e-5):
    """BatchNorm-style loss to match per-feature mean and variance.

    This function is robust to different activation shapes. It treats a
    "channel" or "feature" dimension as the normalization axis and
    computes per-feature mean and variance across the remaining
    dimensions. Common supported shapes include (B, C, H, W) and
    (B, N, D).

    Args:
        x (torch.Tensor): activation tensor.
        target_mean (float or torch.Tensor): scalar or per-feature
            target mean(s).
        target_var (float or torch.Tensor): scalar or per-feature
            target variance(s).
        eps (float): small value to stabilize numerical ops (unused but
            present to match common BN signatures).

    Returns:
        torch.Tensor: scalar tensor containing mean_loss + var_loss.
    """
    # Move the feature/channel dimension to dim=1 and flatten the rest
    if x.dim() == 4:
        # (B, C, H, W) -> (C, B*H*W)
        x_reshaped = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    else:
        # Move last dim (typical for LayerNorm outputs) to position 1
        x_moved = x.movedim(-1, 1)
        x_reshaped = x_moved.reshape(x_moved.shape[1], -1)

    mean = x_reshaped.mean(dim=1)                      # shape (C,)
    var = x_reshaped.var(dim=1, unbiased=False)        # shape (C,)

    # Ensure targets are tensors with correct shape
    if not torch.is_tensor(target_mean):
        target_mean = x_reshaped.new_full((mean.shape[0],), float(target_mean))
    if not torch.is_tensor(target_var):
        target_var = x_reshaped.new_full((var.shape[0],), float(target_var))

    # If targets are scalars expanded to per-feature, broadcasting will work
    mean_loss = torch.mean((mean - target_mean) ** 2)
    var_loss = torch.mean((var - target_var) ** 2)

    return mean_loss + var_loss



def activation_errors(model, x1, x2):
    """Compute activation-level error metrics for every module in the network."""
    model.eval()

    device = next(model.parameters()).device

    hooks = []
    data = defaultdict(dict)
    inputs = torch.cat((x1, x2), dim=0)
    separator = x1.shape[0]

    def check_activations(self, input, output):
        module_name = str(*[name for name, mod in model.named_modules() if self is mod])
        try:
            layer_inputs = input[0].detach()
            residual = (layer_inputs[:separator] - layer_inputs[separator:]).pow(2)
            se_error = residual.sum()
            mse_error = residual.mean()
            sim = torch.nn.functional.cosine_similarity(layer_inputs[:separator].flatten(),
                                                        layer_inputs[separator:].flatten(),
                                                        dim=0, eps=1e-8).detach()
            data['se'][module_name] = se_error.item()
            data['mse'][module_name] = mse_error.item()
            data['sim'][module_name] = sim.item()
        except (KeyboardInterrupt, SystemExit):
            raise
        except AttributeError:
            pass

    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(check_activations))

    try:
        outputs = model(inputs.to(device))
        for hook in hooks:
            hook.remove()
    except Exception as e:
        for hook in hooks:
            hook.remove()
        raise

    return data
