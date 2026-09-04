MECHANISM: Per-channel learned max–average downsampling

HYPOTHESIS: Learned per-channel pooling, paired with the qualified symmetric ensemble, will exceed 9,215 correct predictions by retaining distributed shape evidence while preserving maxima for edge-selective channels.

INTENDED_EDIT: Replace fixed max pooling in every residual stage with a trainable 75%-max/25%-average blend per output channel, adding only 192 parameters, and use Reference Design 3’s best-verified symmetric inference and calibration.

EVIDENCE: Reference Design 3 improved from 9,210 to 9,215 correct solely through translation-symmetric inference, showing shift sensitivity remains load-bearing. Head widening and channel recalibration regressed, so this tests a different assumption: that irreversible hard-max downsampling is optimal for every learned feature channel.

<<<<<<< SEARCH
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))
=======
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool_mix_logit = nn.Parameter(
            torch.full((1, out_channels, 1, 1), math.log(3.0))
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        features = F.silu(features + refinement, inplace=True)
        max_pooled = F.max_pool2d(features, 2)
        average_pooled = F.avg_pool2d(features, 2)
        max_weight = self.pool_mix_logit.sigmoid()
        return max_pooled * max_weight + average_pooled * (1.0 - max_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        shifted_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
        )
        for view in shifted_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.1875)
=======
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        vertical_views = (
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
        )
        for view in vertical_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        horizontal_views = (
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        for view in horizontal_views:
            view_logits = self._forward_once(view)
            flipped_logits = self._forward_once(view.flip(-1))
            probability_sum.add_(
                F.softmax(view_logits * 1.10, dim=1),
                alpha=0.5,
            )
            probability_sum.add_(
                F.softmax(flipped_logits * 1.10, dim=1),
                alpha=0.5,
            )

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.178)
>>>>>>> REPLACE