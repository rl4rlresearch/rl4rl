MECHANISM: Attention value-bias absorption into the affine output projection

HYPOTHESIS: Removing the redundant eight-dimensional value bias will retain at least 99% accuracy after 45,000 updates while reducing the qualified model from 1,360 to 1,352 learned parameters.

INTENDED_EDIT: Keep only the learned query bias in QKV; set both key and value biases to zero because the output-projection bias can represent the value bias exactly.

EVIDENCE: The current rank-six learned-position model achieved 100% accuracy, and Reference Design 2 achieved 99.69% with value bias removed; this targets a proven exact redundancy without constraining the successful token or positional representations.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # All key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(d_model),
                self.qkv.bias[d_model:],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
>>>>>>> REPLACE