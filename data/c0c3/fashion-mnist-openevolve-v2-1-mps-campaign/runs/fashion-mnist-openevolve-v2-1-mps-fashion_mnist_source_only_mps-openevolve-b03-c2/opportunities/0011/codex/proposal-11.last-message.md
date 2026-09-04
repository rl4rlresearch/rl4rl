MECHANISM: Sparse late-training EMA with batched flip inference

HYPOTHESIS: Updating a 0.995-timescale EMA every eight steps during the second half will preserve most of the observed 9,241-correct EMA benefit while reducing its overhead enough to finish verification and exceed 9,214 correct predictions.

INTENDED_EDIT: Adopt the proven batch-32 flip-paired ten-view design, batch each inference flip pair into one forward call, and install a fused parameter EMA sampled every eight late-training steps.

EVIDENCE: Exact per-step EMA reached 9,241 correct but timed out at 82.89 seconds; the non-EMA design finished with 9,214 correct at 81.30 seconds, while fused per-step EMA still timed out, motivating a large reduction in EMA update frequency.

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
            paired = torch.cat((view, view.flip(-1)), dim=0)
            log_probs = F.log_softmax(self._forward_once(paired), dim=1)
            original, flipped = log_probs.chunk(2, dim=0)
            log_prob_views.extend((original, flipped))
        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=2.5e-3,
        betas=(0.9, 0.99),
=======
        lr=2.0e-3,
        betas=(0.9, 0.99),
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
    completed = step + 1
    ema_start = max(total_steps // 2, 1)
    if completed >= ema_start:
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if not hasattr(optimizer, "_ema_parameters"):
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
            optimizer._ema_last_step = completed

        gap = completed - optimizer._ema_last_step
        if gap >= 8 or completed >= total_steps:
            decay = 0.995 ** gap
            with torch.no_grad():
                torch._foreach_lerp_(
                    optimizer._ema_parameters,
                    parameters,
                    1.0 - decay,
                )
            optimizer._ema_last_step = completed

        if completed >= total_steps:
            with torch.no_grad():
                torch._foreach_copy_(
                    parameters,
                    optimizer._ema_parameters,
                )

    progress = min(max(completed / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE