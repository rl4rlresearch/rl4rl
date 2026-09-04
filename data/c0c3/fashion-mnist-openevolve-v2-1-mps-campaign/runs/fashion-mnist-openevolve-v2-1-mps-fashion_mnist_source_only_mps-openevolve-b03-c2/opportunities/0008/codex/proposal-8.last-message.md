MECHANISM: Full-exposure inference-orbit supervision

HYPOTHESIS: Cycling through the five ten-view spatial transformations during all 100,000 training exposures, while retaining flip-paired supervision, will exceed 9,214 correct predictions by eliminating the clean-half translation mismatch without materially increasing computation.

INTENDED_EDIT: Adopt the proven batch-32 residual model with ten-view inference and flip-paired loss, then extend its center/cardinal one-pixel translation curriculum from the first half to the entire training run.

EVIDENCE: Flip supervision throughout training improved the same model from 9,142 to 9,208 correct, and matching training crops to the five inference views raised this to 9,214; extending that successful alignment through the second half is the direct next test.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0
=======
BATCH_SIZE = 32
GRAD_CLIP_NORM = 5.0
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
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
=======
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step < total_steps // 2:
        padded = F.pad(images, (2, 2, 2, 2), mode="replicate")
        top = step % 5
        left = (step // 5) % 5
        images = padded[:, :, top : top + 28, left : left + 28]
        flip_mask = torch.rand(
            (images.shape[0], 1, 1, 1), device=images.device
        ) < 0.5
        images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
=======
    del total_steps
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
    view = step % 5
    top = (1, 0, 2, 1, 1)[view]
    left = (1, 1, 1, 0, 2)[view]
    images = padded[:, :, top : top + 28, left : left + 28]
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
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