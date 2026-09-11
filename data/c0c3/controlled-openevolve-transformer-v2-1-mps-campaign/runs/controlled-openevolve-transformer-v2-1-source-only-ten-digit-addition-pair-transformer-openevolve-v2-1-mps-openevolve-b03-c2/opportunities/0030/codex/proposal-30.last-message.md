MECHANISM: Six-coordinate value-bias absorption into the attention projection bias

HYPOTHESIS: Extending the verified five-coordinate value-bias quotient to six omitted coordinates will produce a 1589-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.

INTENDED_EDIT: Store two learned value-bias coordinates, reconstruct six zero-gauge coordinates, and absorb all six independently tracked updates into `attn.proj.bias`.

EVIDENCE: Omitting five value-bias coordinates achieved 99.95% accuracy at 1590 parameters, after one through four omissions also exceeded 99%; the sixth coordinate has the same attention-invariant computational role.

<<<<<<< SEARCH
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but three value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 3))
=======
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but six value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(3))
        )
=======
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(6))
        )
>>>>>>> REPLACE