MECHANISM: Two-coordinate softmax-invariant key-bias removal

HYPOTHESIS: Removing a second key-projection bias coordinate while preserving the original layer-construction RNG trajectory will reduce the model from 1,619 to 1,618 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Store 22 learned QKV bias values and reconstruct two fixed-zero key-bias coordinates during attention.

EVIDENCE: Removing one key-bias coordinate achieved 99.74% accuracy at 1,619 parameters, while removing all eight reached only 95.99%; removing one additional coordinate is the smallest informative step toward locating the constraint boundary.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 1))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (self.qkv.bias[:d_model], self.qkv.bias.new_zeros(1), self.qkv.bias[d_model:])
        )
=======
        qkv_bias = torch.cat(
            (self.qkv.bias[:d_model], self.qkv.bias.new_zeros(2), self.qkv.bias[d_model:])
        )
>>>>>>> REPLACE