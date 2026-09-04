MECHANISM: Identity-initialized multi-scale channel attention

HYPOTHESIS: Input-conditioned channel recalibration at all three feature scales will exceed 9,318 correct predictions by learning multiplicative feature interactions that fixed feed-forward and residual representations cannot express.

INTENDED_EDIT: Insert lightweight squeeze-excitation gates before every downsampling stage and narrow the classifier from 147 to 145 hidden units, keeping the model under the parameter ceiling at 249,843 parameters.

EVIDENCE: Residual refinement reached only 9,295 correct and border-complete adaptive max pooling reached 9,314, suggesting that additive feature refinement and pooling geometry are not the missing mechanism; identity-initialized channel attention instead lets global image context dynamically select features while preserving the successful baseline at initialization.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
GRAD_CLIP_NORM = 1.0


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 16, 2)
        self.reduce = nn.Conv2d(channels, hidden, kernel_size=1)
        self.expand = nn.Conv2d(hidden, channels, kernel_size=1)
        nn.init.zeros_(self.expand.weight)
        nn.init.zeros_(self.expand.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = F.adaptive_avg_pool2d(features, 1)
        attention = F.gelu(self.reduce(context))
        attention = 2.0 * torch.sigmoid(self.expand(attention))
        return features * attention


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ChannelAttention(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            ChannelAttention(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
=======
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ChannelAttention(96),
            nn.MaxPool2d(2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
=======
            nn.Linear(96 * 3 * 3, 145),
            nn.BatchNorm1d(145),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(145, 10),
>>>>>>> REPLACE