MECHANISM: Softmax-invariant key-bias elimination

HYPOTHESIS: Removing all eight key-projection bias parameters will reduce the model from 1,643 to 1,635 parameters while retaining at least 99% accuracy, because adding the same learned key bias to every attended position changes each query’s attention logits by a row-wise constant that softmax cancels exactly.

INTENDED_EDIT: Preserve constructor RNG consumption, store only the eight query biases and seven retained value biases, and reconstruct the key bias and final value-bias coordinate as fixed zeros during projection.

EVIDENCE: Single-coordinate value-bias pruning retained 99.96% accuracy, while removing a second value coordinate failed; key bias is a more informative target because, unlike value bias, it has no effect on the attention output even before considering downstream parameters.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the constructor's RNG consumption, then prune one value-bias coordinate.
        self.qkv.bias = nn.Parameter(self.qkv.bias.detach()[:-1].clone())
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve constructor RNG consumption, then remove the softmax-invariant
        # key bias and one value-bias coordinate.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model : -1])).clone()
        )
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = F.linear(x, self.qkv.weight) + F.pad(self.qkv.bias, (0, 1))
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias[:d_model]
        v_bias = F.pad(self.qkv.bias[d_model:], (0, 1))
        bias = torch.cat((q_bias, self.qkv.bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + bias
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE