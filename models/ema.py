from copy import deepcopy

import torch


class ModelEMA(object):
    def __init__(self, args, model, decay, mode='true_avg'):
        assert(mode in ['true_avg', 'run_avg'])
        self.mode = mode
        if self.mode == 'true_avg':
            self.step = 0

        self.model = deepcopy(model)
        self.model.to(args.device)
        self.model.eval()
        self.decay = decay
        self.model_has_module = hasattr(self.model, 'module')
        # Fix EMA. https://github.com/valencebond/FixMatch_pytorch thank you!
        self.param_keys = [k for k, _ in self.model.named_parameters()]
        self.buffer_keys = [k for k, _ in self.model.named_buffers()]
        for p in self.model.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.model_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.model.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                if self.mode == 'true_avg':
                    # true average until sufficient iterations (mean teacher implementation): https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/main.py#L52
                    # decay in mean teacher is also 0.999: https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/mean_teacher/cli.py#L63
                    decay = min(1 - 1 / (self.step + 1), self.decay) 
                    # step = 0, decay = 0; step = \infty, decay = self.decay
                elif self.mode == 'run_avg':
                    decay = self.decay
                esd[k].copy_(ema_v * decay + (1. - decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
        
        if self.mode == 'true_avg':
            self.step += 1
