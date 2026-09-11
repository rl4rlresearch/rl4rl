MECHANISM: Five-coordinate value-bias absorption into the attention projection bias

HYPOTHESIS: Extending the verified four-coordinate value-bias quotient to five omitted coordinates will produce a 1590-parameter model with at least 99% accuracy while preserving clipping and AdamW dynamics for the additional coordinate.

INTENDED_EDIT: Store three learned value-bias coordinates, reconstruct five zero-gauge coordinates, and absorb all five independently tracked AdamW updates into `attn.proj.bias`.

EVIDENCE: Successive omission of one, two, three, and four value-bias coordinates achieved 99.73%, 99.98%, 100%, and 99.92% accuracy respectively; the fifth coordinate has the same attention-invariant computational role.

<<<<<<< SEARCH
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but four value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 4))
=======
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but five value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(4))
        )
=======
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(5))
        )
>>>>>>> REPLACE