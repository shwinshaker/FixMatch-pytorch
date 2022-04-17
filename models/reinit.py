from copy import deepcopy
import torch

from .util import create_model

import logging
logger = logging.getLogger(__name__)

def rand_mask(weight, sparsity):
    mask = torch.zeros(weight.numel(), dtype=bool)
    mask[:int(sparsity * mask.numel())] = True
    idx = torch.randperm(mask.numel())
    mask = mask[idx].view(weight.size())
    return mask


def min_abs_mask(weight, sparsity, dtype=bool):
    q = torch.quantile(torch.abs(weight), q=sparsity)
    if dtype == bool:
        return torch.where(torch.abs(weight) < q, True, False)
    elif dtype == int:
        return torch.where(torch.abs(weight) < q, 1, 0)
    else:
        raise TypeError(dtype)


class Interpolator:
    def __init__(self, args, model, reinit_retain_rate=0.9):
        self.args = args
        self.min_rate = reinit_retain_rate
        self.rate = reinit_retain_rate

        self.layers = {}
        self.masks = {}
        for name, weight in model.named_parameters():
            if 'conv' in name or 'fc' in name:
                self.layers[name] = weight
                assert(weight.requires_grad), name
                # self.masks[name] = torch.ones_like(weight, dtype=bool)
                # self.masks[name] = rand_mask(weight, sparsity=1 - self.rate)

    def update_masks(self):
        for name in self.layers:
            self.masks[name] = min_abs_mask(self.layers[name], 1 - self.rate, dtype=int)

    def check_masks(self):
        # sanity check
        sum_masked = 0
        num_mask = 0
        for name in self.masks:
            sum_masked += self.masks[name].sum()
            num_mask += self.masks[name].numel()
        logger.info('Mask updated with sparsity {:.3f}'.format(sum_masked.item() / num_mask))

    def update(self, model, epoch, batch_idx):
        if not batch_idx + 1 == self.args.steps_per_epoch:
            # only update once after the end of each epoch
            return

        self.update_masks()
        self.check_masks()

        model_init = create_model(self.args)
        model_init.to(self.args.device)
        model_init_state_dict = model_init.state_dict() 

        with torch.no_grad():
            for name in self.layers:
                self.layers[name].copy_(self.layers[name] * (1 - self.masks[name]) + model_init_state_dict[name] * self.masks[name])
        logger.info('  Model reinitialized with retain rate {:.8g}'.format(self.rate))

        # gradually increasing retain rate through training
        self.rate = self.min_rate + (1 - self.min_rate)* (epoch + 1) / self.args.epochs

        # logger.info('  Model reinitialized with interpolation with retain rate {:.8g}'.format(self.rate))
        # param_keys = [k for k, _ in model_init.named_parameters()]
        # # buffer_keys = [k for k, _ in model_init.named_buffers()] # buffer unchanged
        # needs_module = hasattr(model, 'module')
        # with torch.no_grad():
        #     msd = model.state_dict()
        #     misd = model_init.state_dict()
        #     for k in param_keys:
        #         if needs_module:
        #             j = 'module.' + k
        #         else:
        #             j = k
        #         msd[j].copy_(msd[j] * self.rate + (1. - self.rate) * misd[k])



class Sparsor:
    # mask smallest K weights
    def __init__(self, args, model, sparsity=0.3):
        
        self.args = args
        self.max_sparsity = sparsity
        self.sparsity = sparsity

        self.layers = {}
        self.masks = {}
        for name, weight in model.named_parameters():
            if 'conv' in name or 'fc' in name:
                self.layers[name] = weight
                assert(weight.requires_grad), name
                # self.masks[name] = torch.ones_like(weight, dtype=bool)
                self.masks[name] = rand_mask(weight, sparsity=self.sparsity)

    def update_masks(self):
        for name in self.layers:
            self.masks[name] = min_abs_mask(self.layers[name], self.sparsity)

    def check_masks(self):
        # sanity check
        sum_masked = 0
        num_mask = 0
        for name in self.masks:
            sum_masked += self.masks[name].sum()
            num_mask += self.masks[name].numel()
        logger.info('Mask updated with sparsity {:.3f}'.format(sum_masked.item() / num_mask))

    def update(self, model, epoch, batch_idx):
        if batch_idx + 1 == self.args.steps_per_epoch:
            self.update_masks()
            self.check_masks()

            # gradually decreasing sparsity during training (thus increasing effective capacity)
            self.sparsity = self.max_sparsity * (1 - (epoch + 1) / self.args.epochs)

        for name in self.layers:
            self.layers[name].data[self.masks[name]] = 0
            self.layers[name].grad.data[self.masks[name]] = 0
            # self.layers[name].data[self.masks[name]] = 0
            # self.layers[name].grad.data[self.masks[name]] = 0

        # for (w, b), (mask_w, mask_b) in zip(self.layers, self.masks):
        #     # Values
        #     w.data[mask_w] = 0
        #     b.data[mask_b] = 0
        #     # Grad
        #     w.grad.data[mask_w] = 0
        #     b.grad.data[mask_b] = 0


            