MECHANISM: Two-coordinate query-bias gauge fixing

HYPOTHESIS: Fixing a second query-bias coordinate at zero will reduce the model from 1,601 to 1,600 parameters while retaining at least 99% accuracy, because the learned pre-attention bias and full value bias can represent query offsets and six explicit query-bias coordinates should preserve most of the optimization benefit.

INTENDED_EDIT: Store six learned query-bias coordinates plus all eight value-bias coordinates, reconstructing the final two query-bias coordinates as zeros during attention.

EVIDENCE: Fixing one query-bias coordinate achieved 99.94% accuracy with 1,601 parameters, while removing all eight still achieved 95.61%; an incremental second-coordinate removal directly tests how much of the remaining query-bias parameterization is optimization-critical.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain seven query-bias coordinates and the full value bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model - 1))
=======
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain six query-bias coordinates and the full value bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (self.qkv.bias[: d_model - 1], self.qkv.bias.new_zeros(1))
        )
        v_bias = self.qkv.bias[d_model - 1 :]
=======
        q_bias = torch.cat(
            (self.qkv.bias[: d_model - 2], self.qkv.bias.new_zeros(2))
        )
        v_bias = self.qkv.bias[d_model - 2 :]
>>>>>>> REPLACE