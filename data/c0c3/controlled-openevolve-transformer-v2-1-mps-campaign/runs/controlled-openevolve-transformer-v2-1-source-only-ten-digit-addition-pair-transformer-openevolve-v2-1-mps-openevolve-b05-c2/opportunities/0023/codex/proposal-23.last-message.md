MECHANISM: Attention value/output-bias folding

HYPOTHESIS: Replacing the redundant value bias plus anchored projection bias with one full mean-zero projection bias will produce a 1574-parameter model with at least 99% accuracy while preserving the initialized function.

INTENDED_EDIT: Remove the eight-coordinate value bias, retain the query bias, and restore all seven direct mean-zero attention output-bias coordinates.

EVIDENCE: Fixing a third direct projection-bias coordinate collapsed accuracy to 74.33%, and deriving it from value bias reached only 71.77%; this indicates that direct output-bias conditioning is important, while the value bias is functionally redundant because causal attention weights sum to one and its constant contribution can be folded into the projection bias.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = AnchoredMeanZeroOutputLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (self.qv_bias[0], self.qkv.weight.new_zeros(d_model), self.qv_bias[1])
        )
=======
        qkv_bias = torch.cat(
            (
                self.q_bias,
                self.qkv.weight.new_zeros(d_model),
                self.qkv.weight.new_zeros(d_model),
            )
        )
>>>>>>> REPLACE