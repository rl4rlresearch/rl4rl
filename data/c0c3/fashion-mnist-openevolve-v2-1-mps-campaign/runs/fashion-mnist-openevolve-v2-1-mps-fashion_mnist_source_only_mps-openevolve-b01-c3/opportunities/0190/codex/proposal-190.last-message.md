MECHANISM: Oriented gradient-lifted convolutional stem

HYPOTHESIS: On the validated uniform-pair baseline, explicitly supplying horizontal and vertical edge magnitudes will exceed 9,325 correct predictions by learning shape-sensitive features more efficiently within 782 optimizer steps.

INTENDED_EDIT: Restore the best uniform 37.5% pairing and hybrid inference control, then replace the raw-intensity-only stem input with intensity plus fixed oriented gradient magnitudes; narrow the MLP to remain below 250,000 parameters.

EVIDENCE: Uniform pairing with hybrid pooling reached 9,325 correct, while later loss and inference refinements did not improve correctness and center anchoring fell to 9,324. This challenges the shared assumption that the shallow stem can discover all useful nonlinear edge representations from raw intensity during the fixed exposure, without repeating the computationally heavier spatial-head designs that timed out.

<<<<<<< SEARCH
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
=======
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(192, 61),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(61, 10),
=======
            nn.Linear(192, 58),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(58, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.early(features))
=======
    def _predict(self, images: torch.Tensor) -> torch.Tensor:
        padded_images = F.pad(
            images,
            (1, 1, 1, 1),
            mode="replicate",
        )
        horizontal_edges = (
            padded_images[:, :, 1:-1, 2:]
            - padded_images[:, :, 1:-1, :-2]
        ).abs()
        vertical_edges = (
            padded_images[:, :, 2:, 1:-1]
            - padded_images[:, :, :-2, 1:-1]
        ).abs()
        stem_input = torch.cat(
            (images, horizontal_edges, vertical_edges),
            dim=1,
        )
        features = self.stem(stem_input)
        features = F.gelu(features + self.early(features))
>>>>>>> REPLACE

<<<<<<< SEARCH
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        pooled_predictions = pooled_logits.argmax(dim=-1)
=======
        pooled_logits = (
            offset_weights.unsqueeze(-1) * offset_logits
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        offset_probabilities = view_logits.softmax(dim=-1).mean(dim=1)
        pooled_probabilities = (
            offset_weights.unsqueeze(-1) * offset_probabilities
        ).sum(dim=0) / offset_weights.sum(dim=0).unsqueeze(1)
        hybrid_probabilities = (
            0.875 * pooled_logits.softmax(dim=-1)
            + 0.125 * pooled_probabilities
        )
        pooled_logits = hybrid_probabilities.clamp_min(1.0e-8).log()
        pooled_predictions = pooled_logits.argmax(dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        calibration = 1.22775 * (
=======
        calibration = 1.24003 * (
>>>>>>> REPLACE

<<<<<<< SEARCH
    positions = torch.randint(0, 5, (batch,), device=images.device)
    random_alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    center_alternate_positions = torch.where(
        positions == 0,
        torch.randint(1, 5, (batch,), device=images.device),
        torch.zeros_like(positions),
    )
    center_anchor_mask = torch.rand(batch, device=images.device) < 0.7137
    alternate_positions = torch.where(
        center_anchor_mask,
        center_alternate_positions,
        random_alternate_positions,
    )
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
=======
    positions = torch.randint(0, 5, (batch,), device=images.device)
    alternate_positions = (
        positions
        + torch.randint(1, 5, (batch,), device=images.device)
    ) % 5
    cross_offset_mask = torch.rand(batch, device=images.device) < 0.375
>>>>>>> REPLACE