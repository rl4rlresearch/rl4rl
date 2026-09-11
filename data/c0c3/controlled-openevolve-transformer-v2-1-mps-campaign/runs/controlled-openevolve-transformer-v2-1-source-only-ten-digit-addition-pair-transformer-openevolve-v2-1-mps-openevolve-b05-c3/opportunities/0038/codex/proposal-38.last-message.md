MECHANISM: Third query-bias coordinate ablation

HYPOTHESIS: Retaining three learned query-bias coordinates will reduce the verified model to 1,598 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reduce the learned query bias from four coordinates to three, leaving the qualified positional, MLP-bias, and single-key-row quotients unchanged.

EVIDENCE: Reducing query bias from five coordinates to four was compatible with 99.91% accuracy at 1,600 parameters, and the current single-key-row quotient remains qualified at 1,599; this isolates one previously untested scalar instead of repeating failed MLP or second-key-row reductions.

<<<<<<< SEARCH
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and four query biases remain.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
=======
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and three query biases remain.
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The learned entries occupy the first four query coordinates. The
        # remaining query coordinates and every key/value bias are fixed at zero.
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
=======
        # The learned entries occupy the first three query coordinates. The
        # remaining query coordinates and every key/value bias are fixed at zero.
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 5))
        )
>>>>>>> REPLACE