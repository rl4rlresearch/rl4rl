MECHANISM: Single-coordinate softmax-invariant key-bias removal

HYPOTHESIS: Removing one key-projection bias coordinate while preserving the original layer-construction RNG trajectory will reduce the model from 1,620 to 1,619 parameters and retain at least 99% accuracy.

INTENDED_EDIT: Store 23 learned QKV bias values and reconstruct one fixed-zero key-bias coordinate during attention.

EVIDENCE: Initialization-preserving removal of all eight mathematically redundant key biases reached 95.99%; removing only one coordinate is the smallest informative constraint and minimizes the optimizer perturbation.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (self.qkv.bias[:d_model], self.qkv.bias.new_zeros(1), self.qkv.bias[d_model:])
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE