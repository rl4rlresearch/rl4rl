MECHANISM: Antithetic cardinal-shift and flip-orbit supervision

HYPOTHESIS: Pairing opposite one-pixel translations with both horizontal orientations during every update will align training with ten-view inference and exceed the prior best of 9,208 correct predictions.

INTENDED_EDIT: Adopt the best batch-32 ten-view residual design, then train each example with its original view plus an alternating pair of opposite horizontal or vertical shifts, including both flips of all three views.

EVIDENCE: Explicit flip-paired supervision raised the same architecture from 9,142 to 9,208 correct; extending that successful symmetry-matching mechanism to the four translations already used at inference is the most direct next test.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 32
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stage1(images)
        features = self.stage2(features)
        features = self.stage3(features)
        features = F.relu(features + self.residual(features), inplace=True)
        return self.classifier(self.pool(features))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stage1(images)
        features = self.stage2(features)
        features = self.stage3(features)
        features = F.relu(features + self.residual(features), inplace=True)
        return self.classifier(self.pool(features))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._forward_once(images)

        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        log_prob_views = []
        for view in views:
            log_prob_views.append(
                F.log_softmax(self._forward_once(view), dim=1)
            )
            log_prob_views.append(
                F.log_softmax(self._forward_once(view.flip(-1)), dim=1)
            )
        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.5e-3,
=======
        lr=2.0e-3,
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    if step % 2 == 0:
        shifted_a = padded[:, :, 0:28, 1:29]
        shifted_b = padded[:, :, 2:30, 1:29]
    else:
        shifted_a = padded[:, :, 1:29, 0:28]
        shifted_b = padded[:, :, 1:29, 2:30]

    views = (images, shifted_a, shifted_b)
    paired_images = torch.cat(
        tuple(view for base in views for view in (base, base.flip(-1))),
        dim=0,
    )
    paired_labels = labels.repeat(6)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 2.5e-3 * multiplier
=======
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE