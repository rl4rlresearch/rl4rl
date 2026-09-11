MECHANISM: RNG-aligned fifth attention-output common-mode gauge

HYPOTHESIS: Constraining the fifth attention-output projection column to zero mean will reduce the model to 1,576 parameters while retaining at least 99% accuracy, because the removed component is invisible to downstream LayerNorm and initialization preserves the verified model’s function and RNG consumption.

INTENDED_EDIT: Extend `AttentionGaugeLinear` from four to five zero-mean columns and initialize the added gauge by projecting a full eight-coordinate baseline draw.

EVIDENCE: The 1,577-parameter design achieves 99.73% while already constraining four attention-output columns; the fifth `ln1` tie fell to 42.85%, motivating an adjacent extension of an established independent gauge.

<<<<<<< SEARCH
    """Linear projection with zero-mean output bias and first four weight columns."""
=======
    """Linear projection with zero-mean output bias and first five weight columns."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :4]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 4:].flatten()))
        )
=======
        full_weight = self.weight.detach()
        leading_weight_coords = basis.transpose(0, 1) @ full_weight[:, :5]
        self.weight = nn.Parameter(
            torch.cat((leading_weight_coords.flatten(), full_weight[:, 5:].flatten()))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        gauge_size = 4 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 4
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 4
        )
=======
        gauge_size = 5 * (self.out_features - 1)
        leading_weight = self.bias_basis @ self.weight[:gauge_size].view(
            self.out_features - 1, 5
        )
        remaining_weight = self.weight[gauge_size:].view(
            self.out_features, self.in_features - 5
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
=======
        if isinstance(module, AttentionGaugeLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
            nn.init.normal_(baseline_weight, mean=0.0, std=0.02)
            baseline_leading = baseline_weight[
                : 4 * (module.out_features - 1)
            ].view(module.out_features - 1, 4)
            baseline_remaining = baseline_weight[
                4 * (module.out_features - 1) :
            ].view(module.out_features, module.in_features - 4)
            fifth_weight_coords = (
                module.bias_basis.transpose(0, 1)
                @ baseline_remaining[:, :1]
            )
            compact_weight = torch.cat(
                (
                    torch.cat(
                        (baseline_leading, fifth_weight_coords), dim=1
                    ).flatten(),
                    baseline_remaining[:, 1:].flatten(),
                )
            )
            with torch.no_grad():
                module.weight.copy_(compact_weight)
                module.bias.zero_()
        elif isinstance(module, OutputAnchoredLinear):
            baseline_weight = module.weight.new_empty(module.weight.numel() + 1)
>>>>>>> REPLACE