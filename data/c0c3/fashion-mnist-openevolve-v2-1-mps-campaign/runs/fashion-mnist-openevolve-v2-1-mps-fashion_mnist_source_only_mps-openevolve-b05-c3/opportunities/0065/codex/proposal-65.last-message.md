MECHANISM: Coverage-complete overlapping final pooling

HYPOTHESIS: Replacing only the final 2×2 max pool with a 3×3 stride-2 max pool will exceed 9,193 correct predictions by covering all of the 7×7 final feature map instead of systematically discarding its last row and column.

INTENDED_EDIT: Parameterize residual-stage pooling and use overlapping 3×3 stride-2 pooling in the third stage while preserving the verified residual architecture, 3×3 classifier input, parameter count, optimizer, curriculum, EMA, and TTA.

EVIDENCE: Stagewise residual refinement produced the best result at 9,193 correct with 245,818 parameters. Its final 2×2 stride-2 pool maps 7×7 to 3×3 while omitting one boundary row and column, so correcting that asymmetric information loss is a targeted parameter-free extension.

<<<<<<< SEARCH
class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
=======
class ResidualStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernel_size: int = 2,
    ) -> None:
        super().__init__()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
=======
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_kernel_size, stride=2)
>>>>>>> REPLACE

<<<<<<< SEARCH
            ResidualStage(32, 64),
            ResidualStage(64, 96),
=======
            ResidualStage(32, 64),
            ResidualStage(64, 96, pool_kernel_size=3),
>>>>>>> REPLACE