MECHANISM: Mean-preserving activation-aware classifier pruning

HYPOTHESIS: Pruning the two classifier inputs with the smallest variance-weighted class-centered sensitivity after conventional training will retain at least 85% validation accuracy while saving another 5,705 inference MACs and seven parameters, with all 21 recurrent steps unchanged.

INTENDED_EDIT: Accumulate classifier-feature moments during training, select two low-sensitivity inputs during evaluation, fold their empirical mean contributions into the classifier bias, and retain reversible seven-relative-logit compression.

EVIDENCE: Post-training pruning of one weight-selected feature preserved 85.03% accuracy, while fixed-coordinate pruning failed; this motivates a minimal second reduction chosen using both learned discriminative weights and observed feature variation.

<<<<<<< SEARCH
        self.register_buffer(
            "classifier_dropped_index",
            torch.tensor(-1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "classifier_dropped_weight",
            torch.zeros(7),
            persistent=False,
        )
=======
        self.register_buffer(
            "classifier_dropped_indices",
            torch.zeros(2, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "classifier_dropped_weight",
            torch.zeros(7, 2),
            persistent=False,
        )
        self.register_buffer(
            "classifier_dropped_mean",
            torch.zeros(2),
            persistent=False,
        )
        self.register_buffer(
            "classifier_feature_sum",
            torch.zeros(114),
            persistent=False,
        )
        self.register_buffer(
            "classifier_feature_square_sum",
            torch.zeros(114),
            persistent=False,
        )
        self.register_buffer(
            "classifier_feature_count",
            torch.zeros(()),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.bias.zero_()
                expanded.weight[:7].index_copy_(
                    1,
                    self.classifier_keep_indices,
                    reference_classifier.weight,
                )
                expanded.weight[
                    :7, self.classifier_dropped_index
                ].copy_(self.classifier_dropped_weight)
                expanded.bias[:7].copy_(reference_classifier.bias)
            self.classifier = expanded
=======
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.bias.zero_()
                expanded.weight[:7].index_copy_(
                    1,
                    self.classifier_keep_indices,
                    reference_classifier.weight,
                )
                expanded.weight[:7].index_copy_(
                    1,
                    self.classifier_dropped_indices,
                    self.classifier_dropped_weight,
                )
                bias_correction = (
                    self.classifier_dropped_weight
                    * self.classifier_dropped_mean.unsqueeze(0)
                ).sum(dim=1)
                expanded.bias[:7].copy_(
                    reference_classifier.bias - bias_correction
                )
            self.classifier = expanded
>>>>>>> REPLACE

<<<<<<< SEARCH
            relative_weight = (
                full_classifier.weight[:7] - full_classifier.weight[7:8]
            )
            centered_weight = full_classifier.weight - full_classifier.weight.mean(
                dim=0, keepdim=True
            )
            dropped_index = centered_weight.square().sum(dim=0).argmin()
            all_indices = torch.arange(
                full_classifier.in_features,
                device=full_classifier.weight.device,
            )
            keep_indices = all_indices[all_indices != dropped_index]
            compressed = nn.Linear(
                full_classifier.in_features - 1, 7, bias=True
            ).to(
                device=full_classifier.weight.device,
                dtype=full_classifier.weight.dtype,
            )
            with torch.no_grad():
                compressed.weight.copy_(
                    relative_weight.index_select(1, keep_indices)
                )
                compressed.bias.copy_(
                    full_classifier.bias[:7] - full_classifier.bias[7]
                )
                self.classifier_keep_indices = keep_indices
                self.classifier_dropped_index = dropped_index
                self.classifier_dropped_weight = relative_weight[
                    :, dropped_index
                ].clone()
            self.classifier = compressed
=======
            relative_weight = (
                full_classifier.weight[:7] - full_classifier.weight[7:8]
            )
            centered_weight = full_classifier.weight - full_classifier.weight.mean(
                dim=0, keepdim=True
            )
            feature_count = self.classifier_feature_count.clamp_min(1.0)
            feature_mean = self.classifier_feature_sum / feature_count
            feature_variance = (
                self.classifier_feature_square_sum / feature_count
                - feature_mean.square()
            ).clamp_min(0.0)
            feature_variance = torch.where(
                self.classifier_feature_count > 0,
                feature_variance,
                torch.ones_like(feature_variance),
            )
            sensitivity = (
                centered_weight.square().sum(dim=0) * feature_variance
            )
            dropped_indices = sensitivity.topk(2, largest=False).indices
            all_indices = torch.arange(
                full_classifier.in_features,
                device=full_classifier.weight.device,
            )
            keep_mask = torch.ones_like(all_indices, dtype=torch.bool)
            keep_mask[dropped_indices] = False
            keep_indices = all_indices[keep_mask]
            dropped_mean = feature_mean.index_select(0, dropped_indices)
            dropped_weight = relative_weight.index_select(
                1, dropped_indices
            )
            compressed = nn.Linear(
                full_classifier.in_features - 2, 7, bias=True
            ).to(
                device=full_classifier.weight.device,
                dtype=full_classifier.weight.dtype,
            )
            with torch.no_grad():
                compressed.weight.copy_(
                    relative_weight.index_select(1, keep_indices)
                )
                bias_correction = (
                    dropped_weight * dropped_mean.unsqueeze(0)
                ).sum(dim=1)
                compressed.bias.copy_(
                    full_classifier.bias[:7]
                    - full_classifier.bias[7]
                    + bias_correction
                )
                self.classifier_keep_indices = keep_indices
                self.classifier_dropped_indices = dropped_indices
                self.classifier_dropped_weight = dropped_weight.clone()
                self.classifier_dropped_mean = dropped_mean.clone()
            self.classifier = compressed
>>>>>>> REPLACE

<<<<<<< SEARCH
        features = torch.cat(
            (mean_output[:, :-2], hidden[:, 0, :]), dim=-1
        )
        if features.shape[-1] != self.classifier.in_features:
=======
        features = torch.cat(
            (mean_output[:, :-2], hidden[:, 0, :]), dim=-1
        )
        if self.training:
            with torch.no_grad():
                detached_features = features.detach().to(
                    dtype=self.classifier_feature_sum.dtype
                )
                self.classifier_feature_sum.add_(
                    detached_features.sum(dim=0)
                )
                self.classifier_feature_square_sum.add_(
                    detached_features.square().sum(dim=0)
                )
                self.classifier_feature_count.add_(features.shape[0])
        if features.shape[-1] != self.classifier.in_features:
>>>>>>> REPLACE