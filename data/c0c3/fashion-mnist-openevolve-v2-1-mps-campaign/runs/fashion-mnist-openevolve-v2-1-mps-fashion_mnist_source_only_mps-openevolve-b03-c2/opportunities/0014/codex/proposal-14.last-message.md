MECHANISM: Compute-reallocated late parameter averaging

HYPOTHESIS: Single-view random flips during the augmented half and flip-paired supervision during the clean half will preserve most of the 9,214-correct design’s symmetry benefit while freeing enough computation for a 0.995-decay late EMA to finish and exceed 9,214 correct predictions.

INTENDED_EDIT: Adopt Reference Design 3, batch each inference flip pair, use random flips instead of duplicated views during the first half, retain paired views during the second half, and install a fused EMA accumulated over the second half.

EVIDENCE: Full flip pairing improved 9,142 to 9,208 correct but added about 9 seconds, while exact EMA reached 9,241 correct but exceeded the time limit by roughly 1.6 seconds; eliminating half of the duplicated training forwards creates substantially more runtime headroom than EMA requires.

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
            paired_logits = self._forward_once(
                torch.cat((view, view.flip(-1)), dim=0)
            )
            logits, flipped_logits = paired_logits.chunk(2, dim=0)
            log_prob_views.append(F.log_softmax(logits, dim=1))
            log_prob_views.append(F.log_softmax(flipped_logits, dim=1))
        stacked = torch.stack(log_prob_views, dim=0)
        return torch.logsumexp(stacked, dim=0) - math.log(len(log_prob_views))
>>>>>>> REPLACE

<<<<<<< SEARCH
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    return torch.optim.AdamW(
        model.parameters(),
        lr=2.5e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
=======
def build_optimizer(model: nn.Module, total_steps: int) -> torch.optim.Optimizer:
    del total_steps
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3,
        betas=(0.9, 0.99),
        weight_decay=3.0e-4,
    )
    optimizer._ema_parameters = None
    return optimizer
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
        flip_mask = torch.rand(
            (images.shape[0], 1, 1, 1), device=images.device
        ) < 0.5
        images = torch.where(flip_mask, images.flip(-1), images)
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    if step < total_steps // 2:
        return F.cross_entropy(
            model(images),
            labels,
            label_smoothing=0.02,
        )

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
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    halfway = total_steps // 2
    with torch.no_grad():
        if step + 1 == halfway:
            optimizer._ema_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
        elif step + 1 > halfway:
            torch._foreach_lerp_(
                optimizer._ema_parameters,
                parameters,
                0.005,
            )
            if step + 1 == total_steps:
                torch._foreach_copy_(
                    parameters,
                    optimizer._ema_parameters,
                )

    progress = min(max((step + 1) / max(total_steps, 1), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    multiplier = 0.10 + 0.90 * cosine
    for group in optimizer.param_groups:
        group["lr"] = 2.0e-3 * multiplier
>>>>>>> REPLACE