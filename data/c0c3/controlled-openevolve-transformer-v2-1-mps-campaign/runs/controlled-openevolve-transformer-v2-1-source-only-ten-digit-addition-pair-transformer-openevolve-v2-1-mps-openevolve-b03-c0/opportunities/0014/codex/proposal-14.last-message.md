MECHANISM: Single-coordinate value/output bias redundancy

HYPOTHESIS: Removing one value-projection bias coordinate alongside the validated missing key-bias coordinate will reduce parameters from 1,619 to 1,618 while retaining at least 99% accuracy, because the attention output-projection bias can learn the equivalent constant offset.

INTENDED_EDIT: Store 22 learned QKV bias values and reconstruct fixed-zero first key and first value bias coordinates, preserving the original linear-layer construction and all other model and training settings.

EVIDENCE: Removing one key-bias coordinate achieved 99.74% accuracy at 1,619 parameters, but removing a second key coordinate fell to 70.85%; testing a distinct one-coordinate affine redundancy is therefore more informative than further restricting the key bias, while retaining the query biases whose wholesale removal previously failed.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (self.qkv.bias[:d_model], self.qkv.bias.new_zeros(1), self.qkv.bias[d_model:])
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : 2 * d_model - 1],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[2 * d_model - 1 :],
            )
        )
>>>>>>> REPLACE