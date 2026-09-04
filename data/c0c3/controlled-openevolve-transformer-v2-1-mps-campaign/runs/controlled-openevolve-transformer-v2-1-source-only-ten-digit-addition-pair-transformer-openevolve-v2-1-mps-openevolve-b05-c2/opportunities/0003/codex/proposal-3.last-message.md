MECHANISM: Softmax-invariant key-bias and output-bias-absorbed value-bias elimination

HYPOTHESIS: Retaining only the QKV query bias will reduce the model to 1628 parameters while maintaining at least 99% accuracy, because key bias cancels in softmax and value bias can be represented exactly by the existing attention output-projection bias.

INTENDED_EDIT: Replace the fused QKV bias with an 8-element learned query bias and fixed-zero key/value biases, preserving all weight matrices, MLP capacity, training, and decoding behavior.

EVIDENCE: Removing key bias alone achieved 99.84% accuracy with 1636 parameters, while reducing MLP width failed at 72.21%; this motivates eliminating another functionally redundant attention bias without reducing nonlinear capacity.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv_bias = torch.cat(
            (self.q_bias, self.qkv.weight.new_zeros(2 * d_model))
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE