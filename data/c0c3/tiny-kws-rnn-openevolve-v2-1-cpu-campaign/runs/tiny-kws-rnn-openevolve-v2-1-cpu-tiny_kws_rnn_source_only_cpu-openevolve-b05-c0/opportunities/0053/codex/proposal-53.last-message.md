MECHANISM: Post-training reference-logit compression

HYPOTHESIS: Training the conventional eight-output classifier and algebraically converting it to seven relative logits only at evaluation will preserve at least 85% accuracy while reducing total inference by 92,910 MACs and parameters by 115.

INTENDED_EDIT: Keep the verified training architecture unchanged, but on evaluation replace its 114-to-8 classifier with an equivalent 114-to-7 classifier whose weights and biases are differences from class eight, then append a fixed zero reference logit.

EVIDENCE: The verified eight-output design achieved 85.03%, while training directly with fixed-reference or Helmert seven-logit parameterizations failed despite preserving the softmax function family; postponing exact compression until after conventional optimization isolates those failures as optimization effects.

<<<<<<< SEARCH
        self.classifier = nn.Linear(114, 8)

    def initial_state(
=======
        self.classifier = nn.Linear(114, 8)

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

    def initial_state(
>>>>>>> REPLACE

<<<<<<< SEARCH
        return self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
=======
        logits = self.classifier(
            torch.cat((mean_output[:, :-2], hidden[:, 0, :]), dim=-1)
        )
        if logits.shape[-1] == 7:
            logits = torch.cat((logits, logits.new_zeros(logits.shape[0], 1)), dim=-1)
        return logits
>>>>>>> REPLACE