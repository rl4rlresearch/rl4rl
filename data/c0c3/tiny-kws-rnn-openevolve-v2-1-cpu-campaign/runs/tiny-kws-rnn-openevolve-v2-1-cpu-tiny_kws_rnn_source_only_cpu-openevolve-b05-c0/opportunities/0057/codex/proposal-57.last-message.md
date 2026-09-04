MECHANISM: Correlation-compensated post-training feature pruning

HYPOTHESIS: Retaining the proven minimum-weight feature removal, then removing one additional classifier feature whose activation is well predicted by a retained feature and folding that prediction into the classifier, will preserve at least 85% accuracy while saving another 5,705 validation MACs and seven parameters.

INTENDED_EDIT: Accumulate classifier-feature covariance during training; at evaluation, retain the proven first pruning choice, select a second feature by residual variance-weighted class sensitivity, and fold its best single-feature linear predictor into the compressed seven-logit classifier.

EVIDENCE: One post-training weight-selected removal achieved 85.03%, while two independently variance-weighted removals with mean correction missed by only one validation example at 84.91%; accounting for feature correlation directly targets the remaining information loss without changing the recurrent path or 21-step schedule.

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
            torch.full((2,), -1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "classifier_dropped_weight",
            torch.zeros(7, 2),
            persistent=False,
        )
        self.register_buffer(
            "classifier_regression_index",
            torch.tensor(-1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "classifier_regression_beta",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "classifier_regression_intercept",
            torch.tensor(0.0),
            persistent=False,
        )
        self.register_buffer(
            "feature_sum",
            torch.zeros(114),
            persistent=False,
        )
        self.register_buffer(
            "feature_outer_sum",
            torch.zeros(114, 114),
            persistent=False,
        )
        self.register_buffer(
            "feature_count",
            torch.tensor(0.0),
            persistent=False,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def train(self, mode: bool = True) -> KeywordGRU:
        if mode and self.classifier.out_features == 7:
            reference_classifier = self.classifier
            expanded = nn.Linear(114, 8, bias=True).to(
                device=reference_classifier.weight.device,
                dtype=reference_classifier.weight.dtype,
            )
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
        elif not mode and self.classifier.out_features == 8:
            full_classifier = self.classifier
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
        return super().train(mode)
=======
    def train(self, mode: bool = True) -> KeywordGRU:
        if mode and self.classifier.out_features == 7:
            reference_classifier = self.classifier
            expanded = nn.Linear(114, 8, bias=True).to(
                device=reference_classifier.weight.device,
                dtype=reference_classifier.weight.dtype,
            )
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.bias.zero_()
                expanded.weight[:7].index_copy_(
                    1,
                    self.classifier_keep_indices,
                    reference_classifier.weight,
                )
                expanded.weight[
                    :7, self.classifier_regression_index
                ].sub_(
                    self.classifier_dropped_weight[:, 1]
                    * self.classifier_regression_beta
                )
                expanded.weight[:7].index_copy_(
                    1,
                    self.classifier_dropped_indices,
                    self.classifier_dropped_weight,
                )
                expanded.bias[:7].copy_(
                    reference_classifier.bias
                    - self.classifier_dropped_weight[:, 1]
                    * self.classifier_regression_intercept
                )
            self.classifier = expanded
        elif not mode and self.classifier.out_features == 8:
            full_classifier = self.classifier
            with torch.no_grad():
                relative_weight = (
                    full_classifier.weight[:7] - full_classifier.weight[7:8]
                )
                relative_bias = (
                    full_classifier.bias[:7] - full_classifier.bias[7]
                )
                centered_weight = (
                    full_classifier.weight
                    - full_classifier.weight.mean(dim=0, keepdim=True)
                )
                weight_sensitivity = centered_weight.square().sum(dim=0)
                first_dropped_index = weight_sensitivity.argmin()

                count = self.feature_count.clamp_min(1.0)
                feature_mean = self.feature_sum / count
                feature_covariance = (
                    self.feature_outer_sum / count
                    - feature_mean.unsqueeze(1) * feature_mean.unsqueeze(0)
                )
                feature_variance = (
                    feature_covariance.diagonal().clamp_min(1.0e-8)
                )
                prediction_gain = (
                    feature_covariance.square()
                    / feature_variance.unsqueeze(0)
                )
                prediction_gain.fill_diagonal_(-1.0)
                prediction_gain[:, first_dropped_index] = -1.0
                best_gain, best_predictor = prediction_gain.max(dim=1)
                residual_variance = (
                    feature_variance - best_gain.clamp_min(0.0)
                ).clamp_min(1.0e-8)
                second_score = weight_sensitivity * residual_variance
                second_score[first_dropped_index] = torch.inf
                second_dropped_index = second_score.argmin()
                regression_index = best_predictor[second_dropped_index]
                regression_beta = (
                    feature_covariance[
                        second_dropped_index, regression_index
                    ]
                    / feature_variance[regression_index]
                )
                regression_intercept = (
                    feature_mean[second_dropped_index]
                    - regression_beta * feature_mean[regression_index]
                )

                dropped_indices = torch.stack(
                    (first_dropped_index, second_dropped_index)
                )
                all_indices = torch.arange(
                    full_classifier.in_features,
                    device=full_classifier.weight.device,
                )
                keep_mask = torch.ones_like(all_indices, dtype=torch.bool)
                keep_mask[dropped_indices] = False
                keep_indices = all_indices[keep_mask]

                adjusted_weight = relative_weight.clone()
                adjusted_weight[:, regression_index].add_(
                    relative_weight[:, second_dropped_index]
                    * regression_beta
                )
                compressed = nn.Linear(
                    full_classifier.in_features - 2, 7, bias=True
                ).to(
                    device=full_classifier.weight.device,
                    dtype=full_classifier.weight.dtype,
                )
                compressed.weight.copy_(
                    adjusted_weight.index_select(1, keep_indices)
                )
                compressed.bias.copy_(
                    relative_bias
                    + relative_weight[:, second_dropped_index]
                    * regression_intercept
                )
                self.classifier_keep_indices = keep_indices
                self.classifier_dropped_indices = dropped_indices
                self.classifier_dropped_weight = relative_weight.index_select(
                    1, dropped_indices
                ).clone()
                self.classifier_regression_index = regression_index.clone()
                self.classifier_regression_beta = regression_beta.clone()
                self.classifier_regression_intercept = (
                    regression_intercept.clone()
                )
            self.classifier = compressed
        return super().train(mode)
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
        if self.training and features.shape[-1] == 114:
            with torch.no_grad():
                observed = features.detach()
                self.feature_sum.add_(observed.sum(dim=0))
                self.feature_outer_sum.add_(
                    (
                        observed.unsqueeze(2)
                        * observed.unsqueeze(1)
                    ).sum(dim=0)
                )
                self.feature_count.add_(observed.shape[0])
        if features.shape[-1] != self.classifier.in_features:
>>>>>>> REPLACE