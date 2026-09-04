MECHANISM: Stage-specific 87.5%-max learnable mixed pooling

HYPOTHESIS: Independent pooling stages initialized at 87.5% max will exceed 9,284 correct predictions by modestly strengthening the anti-aliasing that made 90%-max pooling successful, while retaining stage specialization.

INTENDED_EDIT: Restore independent scalar pooling gates and lower their initialization from 90% to 87.5% max pooling; preserve all other settings.

EVIDENCE: Independent 90%-max pooling achieved the best 9,284 correct, whereas moving toward hard max with 95% fell to 9,268 and sharing the gate fell to 9,279, motivating a local search in the opposite direction without sacrificing stage independence.

<<<<<<< SEARCH
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))
=======
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(1.9459101))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
>>>>>>> REPLACE