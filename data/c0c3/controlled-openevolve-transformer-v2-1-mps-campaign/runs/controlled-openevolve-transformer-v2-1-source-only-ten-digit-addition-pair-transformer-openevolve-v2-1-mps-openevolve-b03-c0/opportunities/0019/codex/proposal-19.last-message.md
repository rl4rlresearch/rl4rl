MECHANISM: Balanced per-head value-bias elimination

HYPOTHESIS: Removing the first value-bias coordinate of the second attention head, instead of a second coordinate from the first head, will reduce the model to 1,617 parameters while retaining at least 99% accuracy by distributing the exact value/output-bias redundancy evenly across heads.

INTENDED_EDIT: Store 21 QKV bias values and reconstruct fixed-zero key coordinate 0 plus value coordinates 0 and `head_dim`, preserving the original layer-construction RNG trajectory and all training settings.

EVIDENCE: Removing one key and one value bias reached 99.96% at 1,618 parameters, while removing the adjacent second value coordinate reached 96.11%; testing the corresponding coordinate in the other head isolates whether the failure was caused by concentrating both value constraints within one head.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : 2 * d_model - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 1 :],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : 2 * d_model - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 1 : 2 * d_model + self.head_dim - 2],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model + self.head_dim - 2 :],
            )
        )
>>>>>>> REPLACE