MECHANISM: Content-conditioned feature canonicalization

HYPOTHESIS: Learning a bounded translation of the 7×7 feature grid before local refinement will exceed 9,349 correct predictions by aligning displaced garment structure while preserving the spatial layout whose removal reduced accuracy to 9,085.

INTENDED_EDIT: Replace the fixed-coordinate refinement block with an identity-initialized, content-conditioned feature aligner followed by the established gated refinement, and use the best qualified 1.3477 crop fusion and calibration.

EVIDENCE: Global pooling fell to 9,085, nonlocal attention reached only 9,325, and parallel local refinement reached only 9,321, indicating that spatial layout matters but additional aggregation or filtering is insufficient; meanwhile translation fusion raised the unchanged classifier to 9,349, motivating learned internal alignment rather than continued fusion tuning.

<<<<<<< SEARCH
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

        gate_channels = max(channels // 9, 1)
        self.gate_down = nn.Conv2d(
            channels,
            gate_channels,
            kernel_size=1,
        )
        self.gate_up = nn.Conv2d(
            gate_channels,
            channels,
            kernel_size=1,
        )
        nn.init.zeros_(self.gate_up.weight)
        nn.init.zeros_(self.gate_up.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        refined = self.depthwise(images)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(refined, output_size=1)
            + F.adaptive_max_pool2d(refined, output_size=1)
        )
        gate = 2.0 * torch.sigmoid(
            self.gate_up(F.gelu(self.gate_down(pooled)))
        )
        return F.gelu(images + gate * refined)
=======
class SpatialRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
            padding_mode="replicate",
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

        gate_channels = max(channels // 10, 1)
        self.gate_down = nn.Conv2d(
            channels,
            gate_channels,
            kernel_size=1,
        )
        self.gate_up = nn.Conv2d(
            gate_channels,
            channels + 2,
            kernel_size=1,
        )
        nn.init.zeros_(self.gate_up.weight)
        nn.init.zeros_(self.gate_up.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = 0.5 * (
            F.adaptive_avg_pool2d(images, output_size=1)
            + F.adaptive_max_pool2d(images, output_size=1)
        )
        controls = self.gate_up(F.gelu(self.gate_down(pooled)))
        gate = 2.0 * torch.sigmoid(controls[:, :self.channels])
        shifts = 0.15 * torch.tanh(
            controls[:, self.channels:].flatten(1)
        )

        zeros = torch.zeros_like(shifts[:, :1])
        ones = torch.ones_like(shifts[:, :1])
        transform = torch.cat(
            (
                ones,
                zeros,
                shifts[:, :1],
                zeros,
                ones,
                shifts[:, 1:],
            ),
            dim=1,
        ).reshape(-1, 2, 3)
        grid = F.affine_grid(
            transform,
            images.shape,
            align_corners=False,
        )
        aligned = F.grid_sample(
            images,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        refined = self.depthwise(aligned)
        refined = F.gelu(self.depthwise_norm(refined))
        refined = self.pointwise_norm(self.pointwise(refined))
        return F.gelu(aligned + gate * refined)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fusion_power = 1.174
=======
        fusion_power = 1.3477
>>>>>>> REPLACE

<<<<<<< SEARCH
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() * 1.32772159576416015625
=======
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log() / 0.753170073032379150390625
>>>>>>> REPLACE