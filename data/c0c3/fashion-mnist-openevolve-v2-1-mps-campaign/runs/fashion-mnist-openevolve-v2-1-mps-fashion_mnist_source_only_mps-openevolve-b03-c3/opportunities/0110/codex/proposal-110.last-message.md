MECHANISM: Relative-position non-local part attention

HYPOTHESIS: Adding a low-rank two-head spatial-attention block will exceed 9,283 correct predictions by learning content-dependent relationships among distant garment parts before spatial aggregation, while the residual scale preserves the strong convolutional baseline during short training.

INTENDED_EDIT: Preserve the max-pooled convolutional representation, spatial classifier, augmentation, and TTA; insert compact relative-position self-attention at the 7×7 feature stage and restore the best verified AdamW beta2 of 0.96.

EVIDENCE: Beta2=0.96 reached 9,283 correct, while further optimizer interpolation, pooling replacement, global channel gating, and head changes failed to improve it. This challenges the shared assumption that fixed local/dilated convolution alone adequately models long-range part relationships without discarding the verified backbone or classifier.

<<<<<<< SEARCH
        self.refine = nn.Sequential(
            nn.Conv2d(
                80, 80, kernel_size=3, padding=1, groups=80, bias=False
            ),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.Conv2d(80, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
        )
        self.aggregate_local = nn.Sequential(
=======
        self.refine = nn.Sequential(
            nn.Conv2d(
                80, 80, kernel_size=3, padding=1, groups=80, bias=False
            ),
            nn.BatchNorm2d(80),
            nn.GELU(),
            nn.Conv2d(80, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
        )

        self.attention_query = nn.Conv2d(80, 8, kernel_size=1, bias=False)
        self.attention_key = nn.Conv2d(80, 8, kernel_size=1, bias=False)
        self.attention_value = nn.Conv2d(80, 16, kernel_size=1, bias=False)
        self.attention_project = nn.Conv2d(16, 80, kernel_size=1, bias=False)
        self.attention_norm = nn.BatchNorm2d(80)
        self.attention_scale = nn.Parameter(torch.tensor(0.10))
        self.attention_relative_bias = nn.Parameter(torch.zeros(2, 13, 13))

        coordinate = torch.arange(7)
        rows = coordinate[:, None].expand(7, 7).reshape(-1)
        columns = coordinate[None, :].expand(7, 7).reshape(-1)
        relative_rows = rows[:, None] - rows[None, :] + 6
        relative_columns = columns[:, None] - columns[None, :] + 6
        self.register_buffer(
            "attention_relative_index",
            relative_rows * 13 + relative_columns,
            persistent=False,
        )

        self.aggregate_local = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = torch.cat(
=======
    def _global_context(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = features.shape
        tokens = height * width

        queries = self.attention_query(features).reshape(
            batch, 2, 4, tokens
        ).transpose(-2, -1)
        keys = self.attention_key(features).reshape(batch, 2, 4, tokens)
        scores = torch.matmul(queries, keys) * 0.5
        relative_bias = self.attention_relative_bias.reshape(2, -1)[
            :, self.attention_relative_index
        ]
        attention = F.softmax(scores + relative_bias.unsqueeze(0), dim=-1)

        values = self.attention_value(features).reshape(
            batch, 2, 8, tokens
        ).transpose(-2, -1)
        context = torch.matmul(attention, values)
        context = context.permute(0, 1, 3, 2).reshape(
            batch, 16, height, width
        )
        return self.attention_norm(self.attention_project(context))

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        features = (
            features
            + self.attention_scale * self._global_context(features)
        )
        features = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
        betas=(0.9, 0.95),
=======
        betas=(0.9, 0.96),
>>>>>>> REPLACE