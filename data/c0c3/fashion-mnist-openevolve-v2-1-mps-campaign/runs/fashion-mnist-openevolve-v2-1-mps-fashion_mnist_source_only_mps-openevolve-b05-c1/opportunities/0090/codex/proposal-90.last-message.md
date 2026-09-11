MECHANISM: Class-part cross-attention pooling

HYPOTHESIS: Replacing the fixed 30-dimensional spatial bottleneck with four learned class-specific queries will exceed 9,345 correct predictions by dynamically selecting discriminative image regions for each class while reducing runtime-sensitive computation.

INTENDED_EDIT: Replace the flattened classifier, handcrafted statistics branch, and diagonal gate with a lightweight cross-attention head that pools the 7×7 feature tokens through 40 learned class-part queries and class-specific readouts.

EVIDENCE: Raw-mean gating plateaued at 9,345 correct, while further post-hoc descriptors either regressed or timed out; this challenges the shared assumption that predictions should flow through one fixed global bottleneck by making spatial aggregation class-conditional and input-dependent.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.token_norm = nn.LayerNorm(64)
        self.key_projection = nn.Linear(64, 32, bias=False)
        self.value_projection = nn.Linear(64, 32, bias=False)
        self.part_queries = nn.Parameter(torch.empty(10 * 4, 32))
        self.part_readouts = nn.Parameter(torch.empty(10 * 4, 32))
        self.part_position = nn.Parameter(torch.zeros(10 * 4, 7 * 7))
        self.part_mix = nn.Parameter(torch.full((10, 4), 0.5))
        self.class_bias = nn.Parameter(torch.zeros(10))
        nn.init.xavier_uniform_(self.part_queries)
        nn.init.xavier_uniform_(self.part_readouts)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        spatial_mean = feature_map.mean(dim=(2, 3))
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
        spatial_max = feature_map.amax(dim=(2, 3))
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + self.recalibration_bias
        )
        gated_feature_map = (
            feature_map * channel_gate[:, :, None, None]
        )
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        statistics = statistics * channel_gate.repeat(1, 3)
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)
        keys = self.key_projection(tokens)
        values = self.value_projection(tokens)

        attention_logits = torch.matmul(
            keys, self.part_queries.t()
        ).transpose(1, 2)
        attention_logits = (
            attention_logits / math.sqrt(keys.shape[-1])
            + self.part_position[None]
        )
        attention = F.softmax(attention_logits, dim=-1)
        part_contexts = torch.matmul(attention, values)
        part_logits = (
            part_contexts * self.part_readouts[None]
        ).sum(dim=-1) / math.sqrt(values.shape[-1])
        part_logits = part_logits.reshape(images.shape[0], 10, 4)
        return (
            part_logits * self.part_mix[None]
        ).sum(dim=-1) + self.class_bias
>>>>>>> REPLACE