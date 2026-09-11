MECHANISM: Class-evidence mapping before spatial pooling

HYPOTHESIS: Replacing the pooled-feature MLP with per-location class-evidence maps will exceed 9,325 correct predictions because it learns channel co-occurrences before spatial reduction instead of assuming independently pooled channels retain all discriminative structure.

INTENDED_EDIT: Restore the validated 37.5% paired offset/flip training control, then replace global feature pooling followed by classification with a lightweight convolutional class head and learned per-class mean/max evidence pooling.

EVIDENCE: Reference Design 2 reached 9,325 correct with 37.5% paired offset exposure. The spatial-attention attempt timed out and only changed location weighting; this cheaper alternative tests the distinct load-bearing assumption that class interactions can occur after spatial information has already been discarded.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.LayerNorm(192),
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
        features = self.down1(self.pool1(features))
        features = self.mid_context(features)
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        features = self.late_refinement(features)
        mean_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        peak_features = F.adaptive_max_pool2d(features, 1).flatten(1)
        return self.classifier(torch.cat((mean_features, peak_features), dim=1))
=======
        self.class_evidence = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 10, kernel_size=1),
        )
        self.class_pool_mix = nn.Parameter(
            torch.full((10,), math.log(1.0 / 3.0))
        )

    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
        features = self.down1(self.pool1(features))
        features = self.mid_context(features)
        features = self.down2(self.pool2(features))
        features = self.late_context(features)
        features = self.late_refinement(features)
        evidence_maps = self.class_evidence(features)
        mean_logits = F.adaptive_avg_pool2d(evidence_maps, 1).flatten(1)
        peak_logits = F.adaptive_max_pool2d(evidence_maps, 1).flatten(1)
        pool_mix = torch.sigmoid(self.class_pool_mix).unsqueeze(0)
        return mean_logits + pool_mix * (peak_logits - mean_logits)
>>>>>>> REPLACE

<<<<<<< SEARCH
        calibration = 1.22775 * (
            0.92211476 + 0.07788524 * view_agreement
        )
=======
        agreement_uncertainty = view_agreement * (1.0 - view_agreement)
        calibration = 1.22775 * (
            0.92211476
            + 0.07788524 * view_agreement
            - 0.02 * agreement_uncertainty
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    row_offsets = torch.tensor(
        (1, 0, 2, 1, 1), device=images.device
    )[positions].unsqueeze(1)
    col_offsets = torch.tensor(
        (1, 1, 1, 0, 2), device=images.device
    )[positions].unsqueeze(1)

    rows = row_offsets + torch.arange(height, device=images.device).unsqueeze(0)
    row_index = rows[:, None, :, None].expand(
        batch, channels, height, padded.shape[-1]
    )
    images = padded.gather(2, row_index)

    cols = col_offsets + torch.arange(width, device=images.device).unsqueeze(0)
    col_index = cols[:, None, None, :].expand(batch, channels, height, width)
    images = images.gather(3, col_index)

    flip_mask = torch.rand(batch, device=images.device) < 0.5
    images = torch.where(
        flip_mask[:, None, None, None],
        images.flip(-1),
        images,
    )
    return images, labels
=======
    del step, total_steps
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
    del step, total_steps
    batch, channels, height, width = images.shape
    padded = F.pad(images, (1, 1, 1, 1), mode="replicate")

    positions = torch.randint(0, 5, (batch,), device=images.device)
    alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
    partner_positions = torch.where(
        cross_offset_mask,
        alternate_positions,
        positions,
    )

    row_choices = torch.tensor(
        (1, 0, 2, 1, 1),
        device=images.device,
    )
    col_choices = torch.tensor(
        (1, 1, 1, 0, 2),
        device=images.device,
    )

    def crop_views(selected_positions: torch.Tensor) -> torch.Tensor:
        row_offsets = row_choices[selected_positions].unsqueeze(1)
        col_offsets = col_choices[selected_positions].unsqueeze(1)
        rows = (
            row_offsets
            + torch.arange(height, device=images.device).unsqueeze(0)
        )
        row_index = rows[:, None, :, None].expand(
            batch,
            channels,
            height,
            padded.shape[-1],
        )
        cropped = padded.gather(2, row_index)
        cols = (
            col_offsets
            + torch.arange(width, device=images.device).unsqueeze(0)
        )
        col_index = cols[:, None, None, :].expand(
            batch,
            channels,
            height,
            width,
        )
        return cropped.gather(3, col_index)

    original_views = crop_views(positions)
    partner_views = crop_views(partner_positions)
    flip_mask = torch.rand(batch, device=images.device) < 0.5
    original_views = torch.where(
        flip_mask[:, None, None, None],
        original_views.flip(-1),
        original_views,
    )
    partner_views = torch.where(
        flip_mask[:, None, None, None],
        partner_views,
        partner_views.flip(-1),
    )

    paired_logits = model(
        torch.cat((original_views, partner_views), dim=0)
    )
    original_logits, partner_logits = paired_logits.chunk(2, dim=0)
    classification_loss = 0.5 * (
        F.cross_entropy(
            original_logits,
            labels,
            label_smoothing=0.02,
        )
        + F.cross_entropy(
            partner_logits,
            labels,
            label_smoothing=0.02,
        )
    )

    original_log_probabilities = F.log_softmax(original_logits, dim=-1)
    partner_log_probabilities = F.log_softmax(partner_logits, dim=-1)
    consistency_loss = 0.5 * (
        F.kl_div(
            original_log_probabilities,
            partner_log_probabilities.exp(),
            reduction="none",
        ).sum(dim=-1)
        + F.kl_div(
            partner_log_probabilities,
            original_log_probabilities.exp(),
            reduction="none",
        ).sum(dim=-1)
    )
    return classification_loss + 0.05 * consistency_loss.mean()
>>>>>>> REPLACE