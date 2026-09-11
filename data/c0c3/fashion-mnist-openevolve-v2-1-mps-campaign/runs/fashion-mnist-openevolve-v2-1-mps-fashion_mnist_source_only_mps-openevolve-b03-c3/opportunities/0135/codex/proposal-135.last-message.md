MECHANISM: Accelerated stage-specific mixed-pooling adaptation

HYPOTHESIS: Giving the two stage-specific 90%-max mixture logits twice the backbone learning rate will exceed 9,284 correct predictions by allowing faster stage specialization within the fixed 522-step budget.

INTENDED_EDIT: Restore the best verified scalar mixed-pooling architecture and beta2=0.96, isolate its two mixture logits in a 4.0e-3 optimizer group, and preserve the established 2.0e-3 cosine schedule for all other parameters.

EVIDENCE: Independent stage-level mixtures reached 9,284 correct, outperforming a shared coefficient at 9,279 and channelwise coefficients at 9,281; this indicates stage-specific adaptation is useful while finer granularity is harmful.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
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
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(80),
            nn.GELU(),
        )
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
    decay, no_decay, pool_logits = [], [], []
    for name, parameter in model.named_parameters():
        if name.endswith("max_logit"):
            pool_logits.append(parameter)
        else:
            (decay if parameter.ndim > 1 else no_decay).append(parameter)
    return torch.optim.AdamW(
        [
            {
                "params": decay,
                "weight_decay": 5e-4,
                "lr": 2.0e-3,
                "peak_lr": 2.0e-3,
            },
            {
                "params": no_decay,
                "weight_decay": 0.0,
                "lr": 2.0e-3,
                "peak_lr": 2.0e-3,
            },
            {
                "params": pool_logits,
                "weight_decay": 0.0,
                "lr": 4.0e-3,
                "peak_lr": 4.0e-3,
            },
        ],
        betas=(0.9, 0.96),
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * multiplier
>>>>>>> REPLACE