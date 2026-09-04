MECHANISM: Softmax-normalized value-bias absorption

HYPOTHESIS: Removing the attention value bias in addition to the proven-redundant key bias will reduce parameters from 1,644 to 1,628 while retaining at least 99% accuracy, because attention weights sum to one and the retained output-projection bias can represent the value bias’s constant contribution.

INTENDED_EDIT: Use a bias-free packed QKV projection with only an explicit learned query bias, while retaining both residual-output biases and `d_ff=12`.

EVIDENCE: Key-bias elimination achieved 99.95% accuracy with 1,636 parameters, while removing residual-output biases reduced accuracy to 75.38%; this patch preserves those critical output biases and removes a different attention bias that is functionally absorbable by the retained projection bias.

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

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE