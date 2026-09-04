MECHANISM: Budget-neutral residual feature refinement with paired-flip supervision

HYPOTHESIS: Reallocating dense-head parameters to a residual 7×7 convolution while retaining Reference Design 3’s paired-flip loss will exceed 9,276 correct predictions.

INTENDED_EDIT: Remove harmful translation augmentation, restore paired-flip deep supervision, and replace classifier width with a budget-neutral residual convolutional refinement stage.

EVIDENCE: Reference Design 3 achieved the best result—9,276 correct—with paired-flip supervision, while translation reduced performance to 9,141; its 216,346 parameters are concentrated in the dense head, motivating a clean reallocation toward spatial feature extraction.

<<<<<<< SEARCH
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(48, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
            nn.MaxPool2d(2),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 36),
            nn.LayerNorm(36),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(36, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    augmented = images.clone()
    flip_mask = torch.rand(images.shape[0], device=images.device) < 0.5
    augmented[flip_mask] = torch.flip(augmented[flip_mask], dims=(-1,))

    padding = 2
    padded = F.pad(
        augmented,
        (padding, padding, padding, padding),
        mode="replicate",
    )
    crop_windows = padded.unfold(2, images.shape[-2], 1).unfold(
        3, images.shape[-1], 1
    )
    offset_y = torch.randint(
        0, 2 * padding + 1, (images.shape[0],), device=images.device
    )
    offset_x = torch.randint(
        0, 2 * padding + 1, (images.shape[0],), device=images.device
    )
    batch_indices = torch.arange(images.shape[0], device=images.device)
    augmented = crop_windows[
        batch_indices, :, offset_y, offset_x
    ].contiguous()
    return augmented, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    paired_images = torch.cat(
        (images, torch.flip(images, dims=(-1,))),
        dim=0,
    )
    paired_logits = model(paired_images)
    original_logits, flipped_logits = paired_logits.chunk(2, dim=0)
    ensemble_logits = 0.5 * (original_logits + flipped_logits)

    ensemble_loss = F.cross_entropy(
        ensemble_logits,
        labels,
        label_smoothing=0.02,
    )
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
    return 0.5 * (ensemble_loss + view_loss)
>>>>>>> REPLACE