MECHANISM: Incremental sharing of softmax-invariant key bias

HYPOTHESIS: Tying one additional key-bias coordinate to the surviving shared value will yield a 1641-parameter model with at least 99% accuracy, because the analogous 1642-parameter shared-bias design achieved 99.91%.

INTENDED_EDIT: Replace the 24-element fused QKV bias with 21 learned values, reconstructing one zero key-bias coordinate and three key coordinates from one shared parameter.

EVIDENCE: Sharing a second redundant key-bias coordinate succeeded at 1642 parameters, whereas fixing two coordinates independently at zero scored only 98.52%; this motivates extending the successful sharing mechanism by one parameter.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Remove three softmax-invariant key-bias degrees while preserving
        # fused-projection construction and a learned shared key-bias value.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 3))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(1),
                self.qkv.bias[d_model : d_model + 1].expand(3),
                self.qkv.bias[d_model + 1 : 2 * d_model - 3],
                self.qkv.bias[2 * d_model - 3 :],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE