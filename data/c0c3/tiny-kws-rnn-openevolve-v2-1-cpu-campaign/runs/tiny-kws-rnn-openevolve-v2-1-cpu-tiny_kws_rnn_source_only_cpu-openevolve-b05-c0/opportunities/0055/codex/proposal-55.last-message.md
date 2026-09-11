MECHANISM: Post-training discriminative feature pruning

HYPOTHESIS: Removing the classifier input with the smallest class-centered weight norm after conventional eight-logit training will retain at least 85% validation accuracy while saving 5,705 inference MACs and seven learned parameters, with all 21 recurrent steps unchanged.

INTENDED_EDIT: During evaluation, select and remove the least-used of the 114 classifier features before exact conversion to seven relative logits; preserve reversible train/eval transitions and index the retained features during classification.

EVIDENCE: Exact post-training reference-logit compression retained 85.03% while directly constrained training failed, showing that compression after conventional optimization is safer; fixed-coordinate pruning failed, motivating a learned, weight-based coordinate choice and the smallest possible additional reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
=======
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
        self.register_buffer(
            "classifier_keep_indices",
            torch.arange(114, dtype=torch.long),
            persistent=False,
        )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    def train(self, mode: bool = True) -> KeywordGRU:
        if mode and self.classifier.out_features == 7:
            reference_classifier = self.classifier
            expanded = nn.Linear(
                reference_classifier.in_features, 8, bias=True
            ).to(
                device=reference_classifier.weight.device,
                dtype=reference_classifier.weight.dtype,
            )
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.bias.zero_()
                expanded.weight[:7].copy_(reference_classifier.weight)
                expanded.bias[:7].copy_(reference_classifier.bias)
            self.classifier = expanded
        elif not mode and self.classifier.out_features == 8:
            full_classifier = self.classifier
            compressed = nn.Linear(
                full_classifier.in_features, 7, bias=True
            ).to(
                device=full_classifier.weight.device,
                dtype=full_classifier.weight.dtype,
            )
            with torch.no_grad():
                compressed.weight.copy_(
                    full_classifier.weight[:7] - full_classifier.weight[7:8]
                )
                compressed.bias.copy_(
                    full_classifier.bias[:7] - full_classifier.bias[7]
                )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
=======
        features = torch.cat(
            (mean_output[:, :-2], hidden[:, 0, :]), dim=-1
        )
        if features.shape[-1] != self.classifier.in_features:
            features = features.index_select(
                -1, self.classifier_keep_indices
            )
        logits = self.classifier(features)
>>>>>>> REPLACE