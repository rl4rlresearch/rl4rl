MECHANISM: Higher-frequency optimization with exposure-matched EMA and validated ten-view inference

HYPOTHESIS: Halving the batch size to 64 will exceed 9,125 correct predictions by providing roughly twice as many updates, while scaling AdamW decay and EMA momentum preserves their behavior per training example.

INTENDED_EDIT: Adopt the best validated equal-weight center/cardinal/flip ensemble, use batch size 64, halve weight decay, and slow EMA updates to retain the original example-based averaging horizon.

EVIDENCE: Equal ten-view inference achieved the best known score at 9,125 correct, while diagonal and center-weighted variants did not improve it; the next informative axis is therefore training optimization, and the existing batch-128 run performs only 782 updates.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        probability_sum = F.softmax(logits, dim=1)
        probability_sum.add_(
            F.softmax(self._forward_once(images.flip(-1)), dim=1)
        )

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in shifted_views:
            probability_sum.add_(F.softmax(self._forward_once(view), dim=1))
            probability_sum.add_(
                F.softmax(self._forward_once(view.flip(-1)), dim=1)
            )

        return (probability_sum / 10.0).clamp_min(1e-8).log()
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight_decay=3e-4,
=======
        weight_decay=1.5e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
                average.lerp_(tensor, 0.01)
=======
                average.lerp_(tensor, 0.005)
>>>>>>> REPLACE