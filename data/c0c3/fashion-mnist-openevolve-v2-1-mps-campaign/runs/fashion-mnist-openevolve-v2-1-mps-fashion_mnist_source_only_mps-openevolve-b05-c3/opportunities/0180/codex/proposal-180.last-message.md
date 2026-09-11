MECHANISM: Hierarchical multi-scale feature fusion

HYPOTHESIS: Exposing global summaries of the first two residual stages directly to the classifier will exceed 9,210 correct predictions by preserving fine texture and mid-level shape evidence that the final 3×3 representation may discard.

INTENDED_EDIT: Replace the final-stage-only head with a parameter-budget-neutral 245,571-parameter multi-scale head, while restoring the best-qualified training sampler and TTA controls.

EVIDENCE: Reference Design 1 reached 9,210 correct, but every qualified design used the same final-stage-only representation and further TTA temperature tuning plateaued or regressed. This challenges that load-bearing architectural assumption through direct learned fusion of early, middle, and final features.

<<<<<<< SEARCH
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
=======
        self.features = nn.ModuleList(
            (
                ResidualStage(1, 32),
                ResidualStage(32, 64),
                ResidualStage(64, 96),
            )
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3 + 32 + 64, 43),
            nn.SiLU(inplace=True),
            nn.Dropout(0.10),
            nn.Linear(43, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = images
        summaries = []
        for stage_index, stage in enumerate(self.features):
            features = stage(features)
            if stage_index < 2:
                summaries.append(features.mean(dim=(2, 3)))
        summaries.append(features.flatten(1))
        return self.classifier(torch.cat(summaries, dim=1))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        probability_sum = F.softmax(logits * 1.10, dim=1) * 0.375
        probability_sum.add_(
            F.softmax(
                self._forward_once(images.flip(-1)) * 1.10,
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
            probability_sum.add_(F.softmax(view_logits * 1.10, dim=1))
            probability_sum.add_(F.softmax(flipped_logits * 1.10, dim=1))

        return (probability_sum / 6.75).clamp_min(1e-8).log().mul_(1.175)
>>>>>>> REPLACE

<<<<<<< SEARCH
        directions = torch.randint(
            0, 5, (images.shape[0],), device=images.device
        )
=======
        directions = torch.randint(
            0, 6, (images.shape[0],), device=images.device
        )
>>>>>>> REPLACE