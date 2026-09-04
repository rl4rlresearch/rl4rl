MECHANISM: Content-adaptive anti-aliased downsampling

HYPOTHESIS: Replacing both phase-sensitive hard max pools with learned channel-wise mixtures of max pooling and trainable low-pass downsampling will exceed 9,360 correct predictions by learning more translation-stable intermediate representations while retaining sharp discriminative responses.

INTENDED_EDIT: Add depthwise low-pass filters initialized to binomial blur kernels, learn a per-channel mixture with ordinary max pooling, and use these modules at both spatial reductions; the expected parameter count is 248,506.

EVIDENCE: Calibration plateaued at 9,360 correct, while redesigned heads fell to 9,326–9,337, indicating the verified prediction head should be preserved. The model nevertheless relies on ten translated evaluation views after two phase-sensitive max-pooling operations, motivating a clean test of learned downsampling rather than another head modification.

<<<<<<< SEARCH
class ResidualBlock(nn.Module):
=======
class AdaptiveAntiAliasedDownsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.low_pass = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=channels,
            bias=False,
        )
        kernel = torch.tensor(
            (
                (1.0, 2.0, 1.0),
                (2.0, 4.0, 2.0),
                (1.0, 2.0, 1.0),
            ),
            dtype=self.low_pass.weight.dtype,
        ).div_(16.0)
        with torch.no_grad():
            self.low_pass.weight.copy_(
                kernel.view(1, 1, 3, 3).expand_as(
                    self.low_pass.weight
                )
            )
        self.max_mix_logits = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        maximum = F.max_pool2d(inputs, kernel_size=2, stride=2)
        low_pass = self.low_pass(inputs)
        maximum_weight = torch.sigmoid(self.max_mix_logits)
        return (
            maximum_weight * maximum
            + (1.0 - maximum_weight) * low_pass
        )


class ResidualBlock(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
            ResidualBlock(32, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 64),
=======
            ResidualBlock(32, 32),
            AdaptiveAntiAliasedDownsample(32),
            ResidualBlock(32, 64),
            AdaptiveAntiAliasedDownsample(64),
            ResidualBlock(64, 64),
>>>>>>> REPLACE