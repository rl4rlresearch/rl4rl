MECHANISM: Ramped final-weight exponential averaging

HYPOTHESIS: A short-horizon EMA of the unchanged hard-maximum model will exceed 9,322 correct predictions by reducing final-step parameter noise while retaining the reliably completed training path.

INTENDED_EDIT: Restore ordinary BatchNorm behavior and use an AdamW subclass that tracks a ramped parameter EMA without affecting training, then installs the averaged weights after the exact final optimizer step.

EVIDENCE: Ordinary-BatchNorm hard-maximum attention reliably finished in 75.3 seconds with 9,320 correct and slightly better cross-entropy than sample-weighted BatchNorm; attention-descriptor changes mostly regressed or timed out, motivating a low-overhead orthogonal change that leaves every training forward pass unchanged.

<<<<<<< SEARCH
def build_model() -> nn.Module:
    return ImageClassifier()


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
=======
def build_model() -> nn.Module:
    return ImageClassifier()


class EMAAdamW(torch.optim.AdamW):
    def __init__(self, params, total_steps: int, **kwargs) -> None:
        super().__init__(params, **kwargs)
        self.total_steps = total_steps
        self.completed_steps = 0
        self.averaged_parameters = [
            parameter.detach().clone()
            for group in self.param_groups
            for parameter in group["params"]
        ]

    def step(self, closure=None):
        loss = super().step(closure)
        self.completed_steps += 1
        decay = min(
            0.99,
            (1.0 + self.completed_steps) / (10.0 + self.completed_steps),
        )
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
        ]
        with torch.no_grad():
            for average, parameter in zip(
                self.averaged_parameters, parameters
            ):
                average.lerp_(parameter, 1.0 - decay)
            if self.completed_steps >= self.total_steps:
                for average, parameter in zip(
                    self.averaged_parameters, parameters
                ):
                    parameter.copy_(average)
        return loss


def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    return EMAAdamW(
        model.parameters(),
        total_steps=total_steps,
        lr=2.5e-3,
        weight_decay=5e-4,
        betas=(0.9, 0.99),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    batch_fraction = images.shape[0] / BATCH_SIZE
    batch_norm_momentum = 1.0 - (1.0 - 0.1) ** batch_fraction
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = batch_norm_momentum

    paired_images = torch.cat(
=======
    del step, total_steps
    paired_images = torch.cat(
>>>>>>> REPLACE