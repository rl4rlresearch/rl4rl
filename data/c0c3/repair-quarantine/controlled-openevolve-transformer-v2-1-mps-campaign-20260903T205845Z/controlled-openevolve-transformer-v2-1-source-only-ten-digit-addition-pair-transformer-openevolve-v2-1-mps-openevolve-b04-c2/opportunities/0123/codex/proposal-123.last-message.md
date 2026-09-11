MECHANISM: Progressive downstream-absorbed LayerNorm scale quotient

HYPOTHESIS: Reproducing the qualified 1,562-parameter design and fixing a third bias-free terminal `ln2.weight` coordinate at one will produce a 1,561-parameter model with at least 99% accuracy, because its effective scaling remains representable by the corresponding unrestricted `fc1` column.

INTENDED_EDIT: Use six learned `ln1.bias` coordinates and four learned `ln2.bias` coordinates, then store only the first five `ln2` scales and reconstruct the final three as constants.

EVIDENCE: Fixing the first terminal `ln2` scale achieved 99.74% accuracy at 1,563 parameters, and fixing the second improved this to 99.93% at 1,562; extending the same successful independent redundancy by one coordinate is the most informative next reduction.

<<<<<<< SEARCH
        # A uniform pre-attention shift is absorbable by the query bias and
        # output-projection bias; key shifts cancel from attention softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
=======
        # Retain the qualified six-coordinate pre-attention bias quotient.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Every constant post-normalization shift is absorbable by the
        # unrestricted fc1 bias. Remove one additional direction while
        # retaining six coordinates for nearly unchanged optimization.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 2))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
=======
        # The compact bias basis leaves the final three channels bias-free.
        # Their scales are absorbable by the corresponding unrestricted fc1
        # columns, so reconstruct those scales as fixed ones.
        self.ln2.weight = nn.Parameter(
            self.ln2.weight[:-3].detach().clone()
        )

        # Retain the qualified four-coordinate post-normalization bias.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 4))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            self.ln2.weight,
            ln2_bias,
            self.ln2.eps,
        )
=======
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        ln2_weight = torch.cat(
            (self.ln2.weight, self.ln2.weight.new_ones(3))
        )
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            ln2_weight,
            ln2_bias,
            self.ln2.eps,
        )
>>>>>>> REPLACE