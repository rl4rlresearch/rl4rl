MECHANISM: Learned classwise logit-space flip fusion

HYPOTHESIS: Restoring the 9,330-correct linear curriculum and learning a class-specific convex weighting of original and flipped logits will exceed 9,330 correct by adapting view trust while preserving the proven logit-space ensemble.

INTENDED_EDIT: Remove harmful translation augmentation and add ten sigmoid-constrained fusion parameters, initialized to reproduce equal averaging and trained through the existing ensemble loss.

EVIDENCE: Reference Design 1 achieved 9,330 correct, whereas ±2 translations fell to 9,222 and predictive-probability fusion fell to 9,297; this restores the winner and retains arithmetic-logit aggregation while making its view weights learnable.

<<<<<<< SEARCH
        super().__init__()
        self.features = nn.Sequential(
=======
        super().__init__()
        self.flip_mix_logits = nn.Parameter(torch.zeros(10))
        self.features = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = 0.5 * (logits + flipped_logits)
        return logits
=======
    def combine_views(
        self,
        original_logits: torch.Tensor,
        flipped_logits: torch.Tensor,
    ) -> torch.Tensor:
        original_weight = torch.sigmoid(self.flip_mix_logits).unsqueeze(0)
        return (
            original_weight * original_logits
            + (1.0 - original_weight) * flipped_logits
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if not self.training:
            flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
            logits = self.combine_views(logits, flipped_logits)
        return logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    padding = 2
    height, width = images.shape[-2:]
    padded = F.pad(
        images,
        (padding, padding, padding, padding),
        mode="replicate",
    )

    batch_size, channels = images.shape[:2]
    top = torch.randint(
        0,
        2 * padding + 1,
        (batch_size,),
        device=images.device,
    )
    left = torch.randint(
        0,
        2 * padding + 1,
        (batch_size,),
        device=images.device,
    )

    rows = torch.arange(height, device=images.device).unsqueeze(0)
    rows = rows + top.unsqueeze(1)
    translated = padded.gather(
        2,
        rows[:, None, :, None].expand(
            batch_size,
            channels,
            height,
            padded.shape[-1],
        ),
    )

    columns = torch.arange(width, device=images.device).unsqueeze(0)
    columns = columns + left.unsqueeze(1)
    translated = translated.gather(
        3,
        columns[:, None, None, :].expand(
            batch_size,
            channels,
            height,
            width,
        ),
    )
    return translated, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    ensemble_logits = 0.5 * (original_logits + flipped_logits)
=======
    ensemble_logits = model.combine_views(original_logits, flipped_logits)
>>>>>>> REPLACE