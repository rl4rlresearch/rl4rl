MECHANISM: Stacked low-resolution bottleneck residual refinement

HYPOTHESIS: Restoring the verified 9,236-correct design and adding a second 24-channel bottleneck residual block at 7×7 resolution will exceed 9,236 correct predictions while remaining under the 250,000-parameter ceiling.

INTENDED_EDIT: Restore batch-128 flip-only training, the 48-unit classifier, flip-ensemble evaluation, and 2.5e-3 schedule from Reference Design 2, then add a second low-resolution bottleneck block for approximately 249,754 total parameters.

EVIDENCE: Reference Design 2 achieved 9,236 correct after its added 32-channel low-resolution residual block improved Reference Design 3 by 22 correct; another narrower residual block isolates whether additional spatial depth continues that gain without changing the successful regimen.

<<<<<<< SEARCH
BATCH_SIZE = 64
=======
BATCH_SIZE = 128
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.stem_norm = nn.BatchNorm2d(32)

        self.block_conv1 = nn.Conv2d(
=======
        self.stem_norm = nn.BatchNorm2d(32)
        self.stem_refine_conv1 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm1 = nn.BatchNorm2d(32)
        self.stem_refine_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.stem_refine_norm2 = nn.BatchNorm2d(32)

        self.block_conv1 = nn.Conv2d(
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.refine_conv1 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False
        )
        self.refine_norm1 = nn.BatchNorm2d(64)
        self.refine_conv2 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False
        )
        self.refine_norm2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
=======
        self.pool = nn.MaxPool2d(2)

        self.deep_conv1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.deep_norm1 = nn.BatchNorm2d(32)
        self.deep_conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.deep_norm2 = nn.BatchNorm2d(32)
        self.deep_conv3 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.deep_norm3 = nn.BatchNorm2d(64)

        self.post_deep_conv1 = nn.Conv2d(64, 24, kernel_size=1, bias=False)
        self.post_deep_norm1 = nn.BatchNorm2d(24)
        self.post_deep_conv2 = nn.Conv2d(
            24, 24, kernel_size=3, padding=1, bias=False
        )
        self.post_deep_norm2 = nn.BatchNorm2d(24)
        self.post_deep_conv3 = nn.Conv2d(24, 64, kernel_size=1, bias=False)
        self.post_deep_norm3 = nn.BatchNorm2d(64)
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(64 * 7 * 7, 32),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(32, 10),
=======
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(p=0.10),
            nn.Linear(48, 10),
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(F.gelu(self.stem_norm(self.stem_conv(images))))

        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = features
        features = F.gelu(self.refine_norm1(self.refine_conv1(features)))
        features = self.refine_norm2(self.refine_conv2(features))
        features = F.gelu(features + residual)
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = F.gelu(self.stem_norm(self.stem_conv(images)))
        residual = features
        features = F.gelu(
            self.stem_refine_norm1(self.stem_refine_conv1(features))
        )
        features = self.stem_refine_norm2(self.stem_refine_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = self.projection(features)
        features = F.gelu(self.block_norm1(self.block_conv1(features)))
        features = self.block_norm2(self.block_conv2(features))
        features = self.pool(F.gelu(features + residual))

        residual = features
        features = F.gelu(self.deep_norm1(self.deep_conv1(features)))
        features = F.gelu(self.deep_norm2(self.deep_conv2(features)))
        features = self.deep_norm3(self.deep_conv3(features))
        features = F.gelu(features + residual)

        residual = features
        features = F.gelu(
            self.post_deep_norm1(self.post_deep_conv1(features))
        )
        features = F.gelu(
            self.post_deep_norm2(self.post_deep_conv2(features))
        )
        features = self.post_deep_norm3(self.post_deep_conv3(features))
        features = F.gelu(features + residual)
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_once(images)
        if self.training:
            return logits

        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr=1.5e-3,
        weight_decay=5e-4,
=======
        lr=2.5e-3,
        weight_decay=5e-4,
>>>>>>> REPLACE

<<<<<<< SEARCH
    )

    padding = 2
    padded = F.pad(images, (padding, padding, padding, padding))
    batch = images.shape[0]
    offset_y = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    offset_x = torch.randint(0, 2 * padding + 1, (batch,), device=images.device)
    batch_index = torch.arange(batch, device=images.device)[:, None, None]
    row_index = (
        torch.arange(images.shape[-2], device=images.device)[None, :, None]
        + offset_y[:, None, None]
    )
    column_index = (
        torch.arange(images.shape[-1], device=images.device)[None, None, :]
        + offset_x[:, None, None]
    )
    images = padded[:, 0][batch_index, row_index, column_index].unsqueeze(1)
    return images, labels
=======
    )
    return images, labels
>>>>>>> REPLACE

<<<<<<< SEARCH
        group["lr"] = 1.5e-3 * multiplier
=======
        group["lr"] = 2.5e-3 * multiplier
>>>>>>> REPLACE