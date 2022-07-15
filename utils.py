import torch


def logmeanexp(log_prob, dim):
    k = log_prob.shape[dim]
    return torch.logsumexp(log_prob, dim=dim) - torch.log(torch.tensor(k, device=log_prob.device))


class EnableOnly:
    """Context manager for switching on and off only part of a model."""

    def __init__(self, model, *params_to_enable):
        self.model = model
        self.params_to_enable = params_to_enable
        self.prev_mode = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            self.prev_mode[name] = param.requires_grad
            param.requires_grad = False
        for params in self.params_to_enable:
            for param in params:
                param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, param in self.model.named_parameters():
            param.requires_grad = self.prev_mode[name]
