MECHANISM: Slow-timescale stage-specific mixed pooling

HYPOTHESIS: Independent 90%-max pooling gates trained at half the backbone learning rate with beta2=0.96 will exceed 9,284 correct predictions by permitting stage specialization while limiting drift from the empirically best initialization.

INTENDED_EDIT: Restore the best verified two-gate mixed-pooling architecture and beta2=0.96, while assigning the pooling logits a 0.5× learning-rate multiplier throughout the cosine schedule.

EVIDENCE: Independent 90%-max gates achieved 9,284 correct, while shared, channelwise, 87.5%, 95%, and asymmetric variants all regressed; this supports preserving stage independence and the 90% operating point while testing more conservative gate optimization.

<<<<<<< SEARCH
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 192
GRAD_CLIP_NORM = 5.0


class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
        maximum = F.max_pool2d(features, kernel_size=2, stride=2)
        average = F.avg_pool2d(features, kernel_size=2, stride=2)
        return max_weight * maximum + (1.0 - max_weight) * average


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
    del total_steps
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.95),
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    decay, no_decay, pooling = [], [], []
    for name, parameter in model.named_parameters():
        if name.endswith("max_logit"):
            pooling.append(parameter)
        elif parameter.ndim > 1:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 5e-4, "lr_scale": 1.0},
            {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
            {"params": pooling, "weight_decay": 0.0, "lr_scale": 0.5},
        ],
        lr=2.0e-3,
        betas=(0.9, 0.96),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * group["lr_scale"] * multiplier
>>>>>>> REPLACE