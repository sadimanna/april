"""Mechanisms for image reconstruction from parameter gradients."""

import torch
from collections import defaultdict, OrderedDict
import logging
from inversefed.nn import MetaMonkey
from .metrics import total_variation as TV
from .metrics import total_variation_patches as TVP
from .metrics import BNLoss
from .metrics import InceptionScore
from .medianfilt import MedianPool2d
from copy import deepcopy

import time, sys

DEFAULT_CONFIG = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss',
                      bn_loss=0.0)

# Patch-gradient options: if `patch_gradients` is True, only the regions
# selected by a mask will be optimized. Configure selection strategy via
# `patch_selection` in {random, row, col, block}.
DEFAULT_CONFIG.update(dict(
    patch_gradients=True,
    grad_patch_size=16,
    grad_patch_count=8,
    grad_patch_positions=None,
    patch_selection='row',
))

def _label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def _validate_config(config):
    for key in DEFAULT_CONFIG.keys():
        if config.get(key) is None:
            config[key] = DEFAULT_CONFIG[key]
    for key in config.keys():
        if DEFAULT_CONFIG.get(key) is None and 'grad_patch' not in key:
            raise ValueError(f'Deprecated key in config dict: {key}!')
    return config


class GradientReconstructor():
    """Instantiate a reconstruction algorithm."""

    def __init__(self, model, mean_std=(0.0, 1.0), config=DEFAULT_CONFIG, num_images=1):
        """Initialize with algorithm setup."""
        self.config = _validate_config(config)
        self.model = model
        model_parameters = [p for p in self.model.parameters() if p.requires_grad]
        param_names = [n for n,p in self.model.named_parameters() if p.requires_grad]
        self.mp = param_names
        # print(model_parameters)
        self.setup = dict(device=next(iter(model_parameters)).device, dtype=next(iter(model_parameters)).dtype)

        self.mean_std = mean_std
        self.num_images = num_images

        if self.config['scoring_choice'] == 'inception':
            self.inception = InceptionScore(batch_size=1, setup=self.setup)

        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.iDLG = True

        self.reconstructed_patches = []

    def reconstruct(self, input_data, labels, img_shape=(3, 32, 32), dryrun=False, eval=True, tol=None):
        """Reconstruct image from gradient."""
        start_time = time.time()
        if eval:
            self.model.eval()


        stats = defaultdict(list)
        x = self._init_images(img_shape)
        scores = torch.zeros(self.config['restarts'])

        if labels is None:
            if self.num_images == 1 and self.iDLG:
                # iDLG trick:
                last_weight_min = torch.argmin(torch.sum(input_data[-2], dim=-1), dim=-1)
                labels = last_weight_min.detach().reshape((1,)).requires_grad_(False)
                logging.info(f"Reconstructed Labels: {labels}")
                self.reconstruct_label = False
            else:
                # DLG label recovery
                # However this also improves conditioning for some LBFGS cases
                self.reconstruct_label = True

                def loss_fn(pred, labels):
                    labels = torch.nn.functional.softmax(labels, dim=-1)
                    return torch.mean(torch.sum(- labels * torch.nn.functional.log_softmax(pred, dim=-1), 1))
                self.loss_fn = loss_fn
        else:
            assert labels.shape[0] == self.num_images
            self.reconstruct_label = False

        try:
            for trial in range(self.config['restarts']):
                x_trial, labels = self._run_trial(x[trial], input_data, labels, dryrun=dryrun)
                # Finalize
                scores[trial] = self._score_trial(x_trial, input_data, labels)
                x[trial] = x_trial
                if tol is not None and scores[trial] <= tol:
                    break
                if dryrun:
                    break
        except KeyboardInterrupt:
            logging.info('Trial procedure manually interruped.')
            pass

        # Choose optimal result:
        if self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            x_optimal, stats = self._average_trials(x, labels, input_data, stats)
        else:
            logging.info('Choosing optimal result ...')
            scores = scores[torch.isfinite(scores)]  # guard against NaN/-Inf scores?
            optimal_index = torch.argmin(scores)
            logging.info(f'Optimal result score: {scores[optimal_index]:2.4f}')
            stats['opt'] = scores[optimal_index].item()
            x_optimal = x[optimal_index]

        logging.info(f'Total time: {time.time()-start_time}.')
        return x_optimal.detach(), stats

    def _init_images(self, img_shape):
        if self.config['init'] == 'randn':
            return torch.randn((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        elif self.config['init'] == 'rand':
            return (torch.rand((self.config['restarts'], self.num_images, *img_shape), **self.setup) - 0.5) * 2
        elif self.config['init'] == 'zeros':
            return torch.zeros((self.config['restarts'], self.num_images, *img_shape), **self.setup)
        else:
            raise ValueError()
        
    def _get_x_y(self, y_p_i, x_p_i):
        y = torch.randint(0, y_p_i, (1,)).item() #max(1, H - patch_size + 1)
        x = torch.randint(0, x_p_i, (1,)).item()
        return (y, x)

    def _run_trial(self, x_trial, input_data, labels, dryrun=False):
        # Option A: global optimization over full image (original behaviour)
        # Option B: mask-based optimization where only selected patches are updated
        use_patch = self.config.get('patch_gradients', False)

        if use_patch:
            # configurable patch parameters (fallbacks provided)
            patch_size = int(self.config.get('grad_patch_size', 16))
            patch_count = int(self.config.get('grad_patch_count', 32))
            positions = self.config.get('grad_patch_positions', None)

            y_patch_indices = x_trial.shape[2] // patch_size
            x_patch_indices = x_trial.shape[3] // patch_size

            total_patches = y_patch_indices * x_patch_indices
            # number of distinct rounds = floor(total / patch_count)
            num_iters = total_patches // patch_count

            device = x_trial.device
            dtype = x_trial.dtype
            _, _, H, W = x_trial.shape

            # delta is the learnable tensor (only masked areas will have effect)
            delta = torch.zeros_like(x_trial, device=device, dtype=dtype, requires_grad=True)
            # union mask of all patches covered across iterations
            mask_all = torch.zeros_like(x_trial)

            # prepare list of all patch coordinates
            all_positions = [(yy, xx) for yy in range(y_patch_indices) for xx in range(x_patch_indices)]

            if positions is not None:
                # allow using provided positions directly (overrides automatic selection)
                all_positions = positions

            selection = self.config.get('patch_selection', 'random')

            if selection == 'random':
                perm = torch.randperm(len(all_positions))
                perm_positions = [all_positions[i] for i in perm.tolist()]
            elif selection == 'row':
                # row-major ordering (top-to-bottom, left-to-right)
                perm_positions = sorted(all_positions, key=lambda t: (t[0], t[1]))
            elif selection == 'col':
                # column-major ordering (left-to-right, top-to-bottom)
                perm_positions = sorted(all_positions, key=lambda t: (t[1], t[0]))
            elif selection == 'block':
                # create contiguous square-ish blocks of patch indices and flatten
                block_side = max(1, int(patch_count ** 0.5))
                blocks = []
                for by in range(0, y_patch_indices, block_side):
                    for bx in range(0, x_patch_indices, block_side):
                        block_positions = []
                        for i in range(block_side):
                            for j in range(block_side):
                                yy = by + i
                                xx = bx + j
                                if yy < y_patch_indices and xx < x_patch_indices:
                                    block_positions.append((yy, xx))
                        if block_positions:
                            blocks.append(block_positions)
                # flatten blocks in scan order
                perm_positions = [p for block in blocks for p in block]
            else:
                raise ValueError(f'Unknown patch_selection: {selection}')

            # If perm_positions shorter than required, repeat it to fill
            if len(perm_positions) < total_patches:
                times = (total_patches + len(perm_positions) - 1) // len(perm_positions)
                perm_positions = (perm_positions * times)[:total_patches]

            for ni in range(num_iters):
                logging.info(f"## Patch optimization iteration for random set {ni+1} / {num_iters} ##")
                group = perm_positions[ni * patch_count: (ni + 1) * patch_count]

                # mask for this group
                mask = torch.zeros_like(x_trial)
                for (y_idx, x_idx) in group:
                    mask[:, :, y_idx * patch_size:(y_idx + 1) * patch_size, x_idx * patch_size:(x_idx + 1) * patch_size] = 1.0
                    mask_all[:, :, y_idx * patch_size:(y_idx + 1) * patch_size, x_idx * patch_size:(x_idx + 1) * patch_size] = 1.0

                # prepare optimizer parameters
                params = [delta]
                if self.reconstruct_label:
                    output_test = self.model(x_trial + delta * mask)
                    labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)
                    params.append(labels)

                if self.config['optim'] == 'adam':
                    optimizer = torch.optim.Adam(params, lr=self.config['lr'])
                elif self.config['optim'] == 'sgd':  # actually gd
                    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, nesterov=True)
                elif self.config['optim'] in ('lbfgs', 'LBFGS'):
                    optimizer = torch.optim.LBFGS(params)
                else:
                    raise ValueError()

                max_iterations = self.config['max_iterations']
                dm, ds = self.mean_std
                if self.config['lr_decay']:
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                    milestones=[max_iterations // 2.667, max_iterations // 1.6], gamma=0.1)

                try:
                    prev_x_trial = (x_trial + delta * mask).clone().detach()

                    for iteration in range(max_iterations):
                        # pass original x_trial and let closure form x_in via delta*mask
                        closure = self._gradient_closure(optimizer, x_trial, input_data, labels, delta=delta, mask=mask)

                        rec_loss = optimizer.step(closure)
                        if self.config['lr_decay']:
                            scheduler.step()

                        with torch.no_grad():
                            # Project into image space for masked updates
                            x_in = x_trial + delta * mask
                            if self.config['boxed']:
                                clamped = torch.max(torch.min(x_in, (1 - dm) / ds), -dm / ds)
                                # only keep clamped values inside mask
                                delta.data = (clamped - x_trial) * mask + delta.data * (1 - mask)

                            if (iteration + 1 == max_iterations) or iteration % 1000 == 0:
                                logging.info(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                            if (iteration + 1) % 100 == 0:
                                if self.config['filter'] == 'none':
                                    pass
                                elif self.config['filter'] == 'median':
                                    x_filtered = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_in)
                                    # only update delta in masked regions
                                    delta.data = (x_filtered - x_trial) * mask + delta.data * (1 - mask)
                                else:
                                    raise ValueError()
                            if iteration % 1000 == 0:
                                x_in = x_trial + delta * mask
                                logging.info(f"Difference between current and previous version: {torch.nn.functional.mse_loss(prev_x_trial, x_in)}")
                                prev_x_trial = x_in.clone().detach()

                        if dryrun:
                            break
                except KeyboardInterrupt:
                    logging.info(f'Recovery interrupted manually in iteration {iteration}!')
                    pass

            # return composed image using union of masks
            return (x_trial + delta * mask_all).detach(), labels

        else:
            x_trial.requires_grad = True
            if self.reconstruct_label:
                output_test = self.model(x_trial)
                labels = torch.randn(output_test.shape[1]).to(**self.setup).requires_grad_(True)

                if self.config['optim'] == 'adam':
                    optimizer = torch.optim.Adam([x_trial, labels], lr=self.config['lr'])
                elif self.config['optim'] == 'sgd':  # actually gd
                    optimizer = torch.optim.SGD([x_trial, labels], lr=self.config['lr'], momentum=0.9, nesterov=True)
                elif self.config['optim'] == 'LBFGS':
                    optimizer = torch.optim.LBFGS([x_trial, labels])
                else:
                    raise ValueError()
            else:
                if self.config['optim'] == 'adam':
                    optimizer = torch.optim.Adam([x_trial], lr=self.config['lr'])
                elif self.config['optim'] == 'sgd':  # actually gd
                    optimizer = torch.optim.SGD([x_trial], lr=0.01, momentum=0.9, nesterov=True)
                elif self.config['optim'] == 'lbfgs':
                    optimizer = torch.optim.LBFGS([x_trial])
                else:
                    raise ValueError()

            max_iterations = self.config['max_iterations']
            dm, ds = self.mean_std
            if self.config['lr_decay']:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=[max_iterations // 2.667, max_iterations // 1.6], gamma=0.1)
                                                                            #  max_iterations // 1.142], gamma=0.1)   # 3/8 5/8 7/8
            try:
                prev_x_trial = x_trial.clone().detach()

                for iteration in range(max_iterations):
                    # pass delta/mask into closure when using patch-based optimization
                    closure = self._gradient_closure(optimizer, x_trial, input_data, labels)

                    rec_loss = optimizer.step(closure)
                    if self.config['lr_decay']:
                        scheduler.step()

                    with torch.no_grad():
                        # Project into image space
                        if self.config['boxed']:
                            x_trial.data = torch.max(torch.min(x_trial, (1 - dm) / ds), -dm / ds)

                        if (iteration + 1 == max_iterations) or iteration % 1000 == 0:
                            logging.info(f'It: {iteration}. Rec. loss: {rec_loss.item():2.4f}.')

                        if (iteration + 1) % 100 == 0:
                            if self.config['filter'] == 'none':
                                pass
                            elif self.config['filter'] == 'median':
                                x_trial.data = MedianPool2d(kernel_size=3, stride=1, padding=1, same=False)(x_trial)
                            else:
                                raise ValueError()
                        if iteration % 1000 == 0:
                            logging.info(f"Difference between current and previous version: {torch.nn.functional.mse_loss(prev_x_trial, x_trial)}")
                            prev_x_trial = x_trial.clone().detach()

                    if dryrun:
                        break
            except KeyboardInterrupt:
                logging.info(f'Recovery interrupted manually in iteration {iteration}!')
                pass
            return x_trial.detach(), labels

    def _gradient_closure(self, optimizer, x_trial, input_gradient, label, delta=None, mask=None):

        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            # If delta+mask are provided, form the effective input `x_in`.
            if delta is None or mask is None:
                x_in = x_trial
            else:
                x_in = x_trial + delta * mask

            loss = self.loss_fn(self.model(x_in), label)
            gradient = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], create_graph=True, retain_graph=True)
            rec_loss = reconstruction_costs([gradient], input_gradient,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TVP(x_in)
                rec_loss += self.config['total_variation'] * TV(x_in)

            ##############################################################
            if self.config['bn_loss'] > 0:
                # Sweep over normalization layers (BatchNorm / LayerNorm),
                # capture their activations for the current x_in via
                # forward hooks, compute per-layer mean/var loss and take
                # the average across layers.
                self.hooks = []
                self.activations = []
                self.modules_with_stats = []

                def _hook(module, inp, out):
                    try:
                        if isinstance(out, (tuple, list)):
                            out_t = out[0]
                        else:
                            out_t = out
                        if isinstance(out_t, torch.Tensor):
                            self.activations.append(out_t)
                            self.modules_with_stats.append(module)
                    except Exception:
                        pass

                for name, module in self.model.named_modules():
                    if isinstance(module, (torch.nn.LayerNorm, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                        self.hooks.append(module.register_forward_hook(_hook))

                try:
                    _ = self.model(x_in)
                finally:
                    for h in self.hooks:
                        h.remove()

                if len(self.activations) == 0:
                    logging.info("Warning: No BN/LN layers with running stats found; ")
                    self.activations = [x_in]
                    self.modules_with_stats = [None]

                bn_losses = []
                for mod, act in zip(self.modules_with_stats, self.activations):
                    t_mean = mod.bias.detach().to(act.device).to(act.dtype) if (mod is not None and hasattr(mod, 'bias')) else 0.0
                    t_var = mod.weight.square().detach().to(act.device).to(act.dtype) if (mod is not None and hasattr(mod, 'weight')) else 1.0
                    bn_losses.append(BNLoss(act, target_mean=t_mean, target_var=t_var))

                rec_loss += self.config['bn_loss'] * (torch.stack(bn_losses).mean())

            rec_loss.backward()
            if self.config['signed']:
                if delta is None:
                    if x_trial.grad is not None:
                        x_trial.grad.sign_()
                else:
                    if delta.grad is not None:
                        delta.grad.sign_()
            return rec_loss

        return closure

    def _score_trial(self, x_trial, input_gradient, label):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            x_trial.grad = None
            loss = self.loss_fn(self.model(x_trial), label)
            gradient = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], create_graph=False)
            return reconstruction_costs([gradient], input_gradient,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'tvp':
            return TVP(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)
        elif self.config['scoring_choice'] in ['pixelmean', 'pixelmedian']:
            return 0.0
        else:
            raise ValueError()

    def _average_trials(self, x, labels, input_data, stats):
        logging.info(f'Computing a combined result via {self.config["scoring_choice"]} ...')
        if self.config['scoring_choice'] == 'pixelmedian':
            x_optimal, _ = x.median(dim=0, keepdims=False)
        elif self.config['scoring_choice'] == 'pixelmean':
            x_optimal = x.mean(dim=0, keepdims=False)

        self.model.zero_grad()
        if self.reconstruct_label:
            labels = self.model(x_optimal).softmax(dim=1)
        loss = self.loss_fn(self.model(x_optimal), labels)
        gradient = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], create_graph=False)
        stats['opt'] = reconstruction_costs([gradient], input_data,
                                            cost_fn=self.config['cost_fn'],
                                            indices=self.config['indices'],
                                            weights=self.config['weights'])
        logging.info(f'Optimal result score: {stats["opt"]:2.4f}')
        return x_optimal, stats



class FedAvgReconstructor(GradientReconstructor):
    """Reconstruct an image from weights after n gradient descent steps."""

    def __init__(self, model, mean_std=(0.0, 1.0), local_steps=2, local_lr=1e-4,
                 config=DEFAULT_CONFIG, num_images=1, use_updates=True, batch_size=0):
        """Initialize with model, (mean, std) and config."""
        super().__init__(model, mean_std, config, num_images)
        self.local_steps = local_steps
        self.local_lr = local_lr
        self.use_updates = use_updates
        self.batch_size = batch_size

    def _gradient_closure(self, optimizer, x_trial, input_parameters, labels):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr,
                                    use_updates=self.use_updates,
                                    batch_size=self.batch_size)
            rec_loss = reconstruction_costs([parameters], input_parameters,
                                            cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                            weights=self.config['weights'])

            if self.config['total_variation'] > 0:
                rec_loss += self.config['total_variation'] * TV(x_trial)
            rec_loss.backward()
            if self.config['signed']:
                x_trial.grad.sign_()
            return rec_loss
        return closure

    def _score_trial(self, x_trial, input_parameters, labels):
        if self.config['scoring_choice'] == 'loss':
            self.model.zero_grad()
            parameters = loss_steps(self.model, x_trial, labels, loss_fn=self.loss_fn,
                                    local_steps=self.local_steps, lr=self.local_lr, use_updates=self.use_updates)
            return reconstruction_costs([parameters], input_parameters,
                                        cost_fn=self.config['cost_fn'], indices=self.config['indices'],
                                        weights=self.config['weights'])
        elif self.config['scoring_choice'] == 'tv':
            return TV(x_trial)
        elif self.config['scoring_choice'] == 'inception':
            # We do not care about diversity here!
            return self.inception(x_trial)


def loss_steps(model, inputs, labels, loss_fn=torch.nn.CrossEntropyLoss(), lr=1e-4, local_steps=4, use_updates=True, batch_size=0):
    """Take a few gradient descent steps to fit the model to the given input."""
    patched_model = MetaMonkey(model)
    if use_updates:
        patched_model_origin = deepcopy(patched_model)
    for i in range(local_steps):
        if batch_size == 0:
            outputs = patched_model(inputs, patched_model.parameters)
            labels_ = labels
        else:
            idx = i % (inputs.shape[0] // batch_size)
            outputs = patched_model(inputs[idx * batch_size:(idx + 1) * batch_size], patched_model.parameters)
            labels_ = labels[idx * batch_size:(idx + 1) * batch_size]
        loss = loss_fn(outputs, labels_).sum()
        grad = torch.autograd.grad(loss, patched_model.parameters.values(),
                                   retain_graph=True, create_graph=True, only_inputs=True)

        patched_model.parameters = OrderedDict((name, param - lr * grad_part)
                                               for ((name, param), grad_part)
                                               in zip(patched_model.parameters.items(), grad))

    if use_updates:
        patched_model.parameters = OrderedDict((name, param - param_origin)
                                               for ((name, param), (name_origin, param_origin))
                                               in zip(patched_model.parameters.items(), patched_model_origin.parameters.items()))
    return list(patched_model.parameters.values())


def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal'):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif indices == 'def':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    else:
        raise ValueError()

    ex = input_gradient[0]
    if weights == 'linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    else:
        weights = input_gradient[0].new_ones(len(input_gradient))

    total_costs = 0
    for trial_gradient in gradients:
        pnorm = [0, 0]
        costs = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)
