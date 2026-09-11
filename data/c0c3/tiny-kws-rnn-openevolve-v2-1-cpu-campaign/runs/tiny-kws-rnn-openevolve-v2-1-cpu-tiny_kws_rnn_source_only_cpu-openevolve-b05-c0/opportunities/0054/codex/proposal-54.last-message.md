MECHANISM: Post-training rank-six discriminative logit factorization

HYPOTHESIS: Truncating only the weakest singular direction of the class-centered classifier will retain at least 85% accuracy while reducing inference by 53,790 MACs, with all recurrent computation and 114 classifier inputs unchanged.

INTENDED_EDIT: Train the conventional eight-logit classifier, then at evaluation replace it with a class-balanced rank-six two-Linear SVD approximation.

EVIDENCE: Exact post-training logit compression retained 85.03% while directly training compressed logits failed; meanwhile additional summary-coordinate pruning lost accuracy, motivating post-training compression of the weakest classifier direction without discarding recurrent features.

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
        if (
            not mode
            and isinstance(self.classifier, nn.Linear)
            and self.classifier.out_features == 8
        ):
            full_classifier = self.classifier
            rank = 6
            input_factor = nn.Linear(
                full_classifier.in_features, rank, bias=False
            ).to(
                device=full_classifier.weight.device,
                dtype=full_classifier.weight.dtype,
            )
            output_factor = nn.Linear(rank, 8, bias=True).to(
                device=full_classifier.weight.device,
                dtype=full_classifier.weight.dtype,
            )
            with torch.no_grad():
                centered_weight = (
                    full_classifier.weight
                    - full_classifier.weight.mean(dim=0, keepdim=True)
                )
                left, singular_values, right = torch.linalg.svd(
                    centered_weight, full_matrices=False
                )
                input_factor.weight.copy_(right[:rank])
                output_factor.weight.copy_(
                    left[:, :rank] * singular_values[:rank].unsqueeze(0)
                )
                output_factor.bias.copy_(full_classifier.bias)
            self.classifier = nn.Sequential(input_factor, output_factor)
        return super().train(mode)
>>>>>>> REPLACE