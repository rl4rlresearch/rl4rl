MECHANISM: Two-point terminal weight averaging

HYPOTHESIS: Averaging the final parameters with a snapshot from 400 updates earlier will capture much of the 9,241-correct EMA benefit while adding negligible overhead, allowing verification to finish and exceed 9,214 correct predictions.

INTENDED_EDIT: Adopt the proven batch-32 residual design with matched cardinal-view training, flip-paired supervision, and ten-view inference, then perform one terminal parameter average with a late-training snapshot.

EVIDENCE: Reference Design 3 finished with 9,214 correct, while per-step 0.995 EMA reached 9,241 correct but timed out at 82.89 seconds; a single snapshot and final interpolation test the same late-weight stabilization with only two parameter-copy operations.

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
        lr=2.5e-3,
=======
        lr=2.0e-3,
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
    if step < total_steps // 2:
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
    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.5e-3 * multiplier
=======
    completed_steps = step + 1
    progress = min(max(completed_steps / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier

    snapshot_step = max(total_steps - 400, 1)
    if completed_steps == snapshot_step:
        optimizer._late_snapshot = [
            parameter.detach().clone()
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
    elif completed_steps >= total_steps:
        snapshot = getattr(optimizer, "_late_snapshot", None)
        if snapshot is not None:
            parameters = [
                parameter
                for group in optimizer.param_groups
                for parameter in group["params"]
            ]
            with torch.no_grad():
                torch._foreach_lerp_(parameters, snapshot, 0.5)
>>>>>>> REPLACE