MECHANISM: Value-bias absorption into the attention output projection

HYPOTHESIS: Removing all eight value-bias parameters will reduce the verified 1,629-parameter model to 1,621 parameters while retaining at least 99% accuracy, because softmax attention rows sum to one and therefore any constant value bias is exactly absorbable by the learned output-projection bias.

INTENDED_EDIT: Store only the learned query bias in `qkv.bias`; reconstruct both key and value biases as zero while retaining the full learned attention weights and output-projection bias.

EVIDENCE: The current design reaches 99.81% at 1,629 parameters, while further positional, LayerNorm, output-bias, and embedding quotients failed. The already successful removal of the equally functionally redundant key bias motivates testing the distinct exact redundancy between value bias and output-projection bias.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from every attention-softmax row. Retain
        # the original parameter slot while storing only query/value biases.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from every attention-softmax row. A value
        # bias also passes unchanged through attention because each row sums
        # to one, so the output-projection bias can absorb it exactly. Retain
        # only the learned query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
=======
        q_bias = self.qkv.bias
        zero_bias = torch.zeros_like(q_bias)
        qkv_bias = torch.cat((q_bias, zero_bias, zero_bias))
>>>>>>> REPLACE