MECHANISM: Late-trajectory exponential weight averaging

HYPOTHESIS: A 0.99-decay EMA of the reliable hard-maximum model will exceed 9,322 correct predictions by reducing late-step parameter noise without changing training examples, learned-parameter count, or augmentation.

INTENDED_EDIT: Maintain a fused EMA of all learned parameters after every optimizer step and install the averaged weights when evaluation begins.

EVIDENCE: The hard-maximum design reliably reaches 9,320 correct in 66.6–75.3 seconds, while attention variants frequently time out and consistency or schedule changes fell to 9,303–9,315; preserving its exact training path while smoothing only the final weights is the clearest untested improvement.

<<<<<<< SEARCH
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)


def build_model() -> nn.Module:
=======
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)

    def train(self, mode: bool = True) -> ImageClassifier:
        if not mode and self.training:
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.copy_(parameter._ema_weight)
        return super().train(mode)


def build_model() -> nn.Module:
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter._ema_weight = parameter.detach().clone()
    return optimizer
>>>>>>> REPLACE

<<<<<<< SEARCH
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
def after_optimizer_step(
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> None:
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    ema_parameters = [parameter._ema_weight for parameter in parameters]
    with torch.no_grad():
        torch._foreach_lerp_(ema_parameters, parameters, 0.01)

    progress = min(step / max(total_steps, 1), 1.0)
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE