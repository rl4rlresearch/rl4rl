MECHANISM: Output-bias absorption of constant value bias

HYPOTHESIS: Removing both the softmax-invariant key bias and the value bias will reduce the model to 1,628 parameters while retaining at least 99% accuracy, because attention weights sum to one, making the value bias a position-independent offset that the learned output-projection bias can represent.

INTENDED_EDIT: Preserve the baseline fused QKV construction and single fused linear call, but retain only the query-bias parameters and synthesize zero key and value bias components.

EVIDENCE: The initialization-preserving fused zero-key-bias design achieved 99.89% accuracy with 1,636 parameters; extending the same successful fused reparameterization to the value bias targets another mathematically redundant eight-parameter component without narrowing the failed MLP bottleneck.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Preserve the baseline projection construction and RNG stream, while
        # retaining only query bias. Key bias cancels in the softmax, and value
        # bias is a constant offset absorbed by the output projection bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
        qkv = F.linear(x, self.qkv.weight, fused_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE