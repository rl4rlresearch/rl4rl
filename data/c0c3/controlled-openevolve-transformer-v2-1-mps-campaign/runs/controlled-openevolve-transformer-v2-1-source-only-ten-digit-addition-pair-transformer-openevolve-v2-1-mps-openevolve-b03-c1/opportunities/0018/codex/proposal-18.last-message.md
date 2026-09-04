MECHANISM: Single-coordinate query-bias gauge fixing

HYPOTHESIS: Fixing one query-bias coordinate at zero will reduce the model from 1,602 to 1,601 parameters while retaining at least 99% accuracy, because the learned pre-attention bias and retained value bias can represent query offsets, and preserving seven query-bias coordinates avoids the optimization collapse seen when all eight were removed.

INTENDED_EDIT: Store seven learned query-bias coordinates plus all eight value-bias coordinates, reconstructing the final query-bias coordinate as zero during attention.

EVIDENCE: Removing all eight query-bias parameters still reached 95.61%, substantially closer to the requirement than removing the value bias or `ln1` bias; this motivates testing the smallest possible one-coordinate reduction while retaining nearly all of the optimization-sensitive query-bias parameterization.

<<<<<<< SEARCH
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain only the query and value portions of its learned bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model))
=======
        # Construct the original affine first to preserve the baseline RNG stream,
        # then retain seven query-bias coordinates and the full value bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(self.qkv.weight.new_zeros(2 * d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qkv.bias.chunk(2)
        q = q + q_bias
        v = v + v_bias
=======
        q_bias = torch.cat(
            (self.qkv.bias[: d_model - 1], self.qkv.bias.new_zeros(1))
        )
        v_bias = self.qkv.bias[d_model - 1 :]
        q = q + q_bias
        v = v + v_bias
>>>>>>> REPLACE