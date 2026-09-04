MECHANISM: Stage-tied learnable max–average pooling

HYPOTHESIS: Sharing one 90%-max mixture coefficient across both pooling stages at the verified 2.0e-3 learning rate will exceed 9,284 correct predictions by retaining the successful anti-aliasing effect while avoiding the harmful pooling-granularity increase seen with channelwise coefficients.

INTENDED_EDIT: Replace both hard max pools with one shared learnable max–average pool and restore the accuracy-optimal 2.0e-3 optimizer and schedule peak.

EVIDENCE: Two stage-specific scalar mixtures achieved the best result of 9,284 correct, whereas expanding them to 72 channelwise coefficients fell to 9,281; this tests whether stronger sharing improves the successful mechanism’s statistical stability.

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
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
    def __init__(self) -> None:
        super().__init__()
        shared_pool = MixedPool2d()
        self.features = nn.Sequential(
            nn.Conv2d(5, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            shared_pool,
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            shared_pool,
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.1e-3,
        betas=(0.9, 0.96),
=======
        lr=2.0e-3,
        betas=(0.9, 0.96),
>>>>>>> REPLACE

<<<<<<< SEARCH
    for group in optimizer.param_groups:
        group["lr"] = 2.1e-3 * multiplier
=======
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE