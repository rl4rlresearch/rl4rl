MECHANISM: Seven-coordinate value-bias absorption into the attention projection bias

HYPOTHESIS: Extending the qualified six-coordinate value-bias quotient to seven omitted coordinates will produce a 1588-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.

INTENDED_EDIT: Retain one learned value-bias coordinate, reconstruct seven zero-gauge coordinates, and absorb all seven independently tracked updates into `attn.proj.bias`.

EVIDENCE: Omitting six value-bias coordinates achieved 99.73% accuracy at 1589 parameters, after omissions of one through five also exceeded 99%; the seventh coordinate has the same attention-invariant computational role.

<<<<<<< SEARCH
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but two value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 2))
=======
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain one value-bias coordinate; the seven omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(2))
        )
=======
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(7))
        )
>>>>>>> REPLACE