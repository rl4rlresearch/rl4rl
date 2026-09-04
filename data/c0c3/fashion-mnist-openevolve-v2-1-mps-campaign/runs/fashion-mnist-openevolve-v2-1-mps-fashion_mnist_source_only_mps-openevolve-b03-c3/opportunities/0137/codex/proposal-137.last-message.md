MECHANISM: Depth-asymmetric mixed-pooling initialization

HYPOTHESIS: Initializing the first pooling stage at 87.5% max and the second at 92.5% max will exceed 9,284 correct predictions by strengthening early anti-aliasing while preserving sharper semantic evidence later, without changing the successful mean pooling bias or parameter count.

INTENDED_EDIT: Parameterize each scalar pooling gate’s initialization and assign complementary stage-specific max weights averaging 90%; preserve all other architecture, optimization, augmentation, loss, schedule, and TTA settings.

EVIDENCE: Independent 90%-max stage gates achieved the best result of 9,284 correct, while tying them fell to 9,279 and moving both together to 87.5% or 95% fell to 9,279 and 9,268; this motivates testing depth-specific differentiation while keeping the aggregate initialization centered on the verified optimum.

<<<<<<< SEARCH
class MixedPool2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_logit = nn.Parameter(torch.tensor(2.1972246))
=======
class MixedPool2d(nn.Module):
    def __init__(self, max_weight: float = 0.90) -> None:
        super().__init__()
        initial_logit = math.log(max_weight / (1.0 - max_weight))
        self.max_logit = nn.Parameter(torch.tensor(initial_logit))
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
=======
            nn.BatchNorm2d(24),
            nn.GELU(),
            MixedPool2d(0.875),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
=======
            nn.BatchNorm2d(48),
            nn.GELU(),
            MixedPool2d(0.925),
            nn.Conv2d(48, 80, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE