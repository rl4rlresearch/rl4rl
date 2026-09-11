MECHANISM: Channel-specific mixed pooling

HYPOTHESIS: Learning an independent max-versus-average blend for each feature channel will exceed 9,284 correct predictions by preserving the validated initial computation while allowing edge, texture, and shape channels to pool differently.

INTENDED_EDIT: Replace each scalar pooling mixture with a broadcast channel-wise mixture, initialize it to the current scalar value, and use the best verified evaluation scale of 1.16727.

EVIDENCE: Content-dependent channel attention reduced accuracy to 9,274, motivating a simpler 70-parameter static channel specialization; Reference Design 2 established 1.16727 as the best verified calibration.

<<<<<<< SEARCH
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
=======
class MixedPool2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(
            torch.full((1, channels, 1, 1), 2.1972246)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        max_weight = torch.sigmoid(self.max_logit)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
            nn.GELU(),
            MixedPool2d(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(48),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.1682 * logits
=======
        return 1.16727 * logits
>>>>>>> REPLACE