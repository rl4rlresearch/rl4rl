MECHANISM: Bilinear channel co-occurrence pooling with spatial moments

HYPOTHESIS: Replacing the absolute-position flattening bottleneck with multiplicative channel co-occurrences and coarse coordinate moments will exceed 9,311 correct predictions by representing shape-part combinations more directly while remaining cheaper than attention.

INTENDED_EDIT: Project the terminal feature map to 64 channels, encode its mean, horizontal and vertical moments, and upper-triangular Gram matrix, then classify the normalized statistics with a 53-unit head; retain the established training procedure and TTA.

EVIDENCE: Translation-aware aggregation produced the best 9,311-correct design, indicating sensitivity to fixed spatial coordinates. Terminal attention supplied no validation metrics because it timed out, so this tests relational representation through explicit bilinear interactions without attention machinery.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 147),
            nn.BatchNorm1d(147),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(147, 10),
        )
=======
        self.projection = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        coordinates = torch.tensor([-1.0, 0.0, 1.0])
        self.register_buffer(
            "_coord_x", coordinates.repeat(3).view(1, 1, 9), persistent=False
        )
        self.register_buffer(
            "_coord_y",
            coordinates.repeat_interleave(3).view(1, 1, 9),
            persistent=False,
        )
        upper_rows, upper_cols = torch.triu_indices(64, 64)
        self.register_buffer("_upper_rows", upper_rows, persistent=False)
        self.register_buffer("_upper_cols", upper_cols, persistent=False)
        self.classifier = nn.Sequential(
            nn.Linear(3 * 64 + 64 * 65 // 2, 53),
            nn.BatchNorm1d(53),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(53, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.0
        for view in views[1:]:
            view_logits = self.classifier(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self.classifier(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.9315)
=======
    def _classify(self, feature_map: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(feature_map).flatten(2)
        mean = tokens.mean(dim=2)
        horizontal_moment = (tokens * self._coord_x).mean(dim=2)
        vertical_moment = (tokens * self._coord_y).mean(dim=2)
        first_order = torch.cat(
            (mean, horizontal_moment, vertical_moment), dim=1
        )
        first_order = F.normalize(first_order, dim=1)

        gram = torch.bmm(tokens, tokens.transpose(1, 2)) / tokens.shape[2]
        gram = torch.sign(gram) * torch.sqrt(gram.abs() + 1e-6)
        second_order = gram[:, self._upper_rows, self._upper_cols]
        second_order = F.normalize(second_order, dim=1)

        representation = torch.cat((first_order, second_order), dim=1)
        return self.classifier(representation)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._classify(self.features(images))
        if self.training:
            return logits

        padded = F.pad(images, (1, 1, 1, 1))
        views = (
            images,
            padded[:, :, 0:28, 1:29],
            padded[:, :, 2:30, 1:29],
            padded[:, :, 1:29, 0:28],
            padded[:, :, 1:29, 2:30],
        )
        ensemble = logits * 2.0
        for view in views[1:]:
            view_logits = self._classify(self.features(view))
            ensemble = ensemble + view_logits
        for view_index, view in enumerate(views):
            flipped_logits = self._classify(self.features(view.flip(-1)))
            if view_index == 0:
                flipped_logits = flipped_logits * 2.0
            ensemble = ensemble + flipped_logits
        return ensemble / (12.0 * 0.9315)
>>>>>>> REPLACE