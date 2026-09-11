MECHANISM: Learned multi-scale spatial-pyramid fusion

HYPOTHESIS: Fusing aligned shallow, intermediate, and deep feature maps before classification will exceed 9,209 correct predictions by recovering fine-scale cues discarded by the shared deepest-only 3×3 representation.

INTENDED_EDIT: Replace the deepest-only head with a parameter-budgeted learned pyramid fusion head, and retain the best verified eight-pass centered/shifted TTA aggregation.

EVIDENCE: Reference Design 2 reached 9,209 correct, while extensive TTA-weight refinements remained within 9,206–9,209; every qualified design still classified only the final feature map, making cross-scale representation the clearest untested bottleneck. The fusion head remains lightweight, unlike the dynamic-channel design that timed out.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualStage(1, 32),
            ResidualStage(32, 64),
            ResidualStage(64, 96),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 48),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        probability_sum = F.softmax(logits * 1.05, dim=1)
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            )
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 10.0).clamp_min(1e-8).log().mul_(1.05)
=======
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = ResidualStage(1, 32)
        self.stage2 = ResidualStage(32, 64)
        self.stage3 = ResidualStage(64, 96)
        self.fusion = nn.Conv2d(
            32 + 64 + 96,
            48,
            kernel_size=1,
            bias=False,
        )
        self.fusion_norm = nn.BatchNorm2d(48)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(48 * 3 * 3, 73),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(73, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        fine = self.stage1(images)
        middle = self.stage2(fine)
        coarse = self.stage3(middle)
        pyramid = torch.cat(
            (
                F.adaptive_avg_pool2d(fine, 3),
                F.adaptive_avg_pool2d(middle, 3),
                coarse,
            ),
            dim=1,
        )
        fused = F.silu(
            self.fusion_norm(self.fusion(pyramid)),
            inplace=True,
        )
        return self.classifier(fused)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        probability_sum = F.softmax(logits * 1.05, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.05,
                dim=1,
            ),
            alpha=0.375,
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.05, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.05, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.10)
>>>>>>> REPLACE