MECHANISM: Two-dimensional attention-routing bottleneck with full-width values

HYPOTHESIS: Decoupling each head’s two-dimensional query/key routing space from its four-dimensional value space will reduce the model from 1,384 to 1,324 parameters while retaining at least 99% accuracy after 52,000 steps; batch size 256 and less frequent validation will keep verification within the time limit.

INTENDED_EDIT: Replace full-width queries and keys with learned two-dimensional per-head address codes while preserving full-width values, residuals, tied embeddings, and the essential one-neuron MLP; halve the training batch and validate every 1,000 steps.

EVIDENCE: The current full-width model reached 99.99%, and a rank-six positional factorization reached 99.98%, indicating that positional routing tolerates a lower-dimensional learned representation. In contrast, removing the one-neuron MLP collapsed accuracy to 73.76%, so this patch preserves the nonlinear and value-processing capacity while testing whether full query/key width is the load-bearing assumption.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = MeanZeroInputLinear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qk_dim = 2
        qk_width = n_head * self.qk_dim
        self.qkv = MeanZeroInputLinear(d_model, 2 * qk_width + d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(qk_width))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        qkv = self.qkv(x)
        qk_width = self.n_head * self.qk_dim
        q, k, v = qkv.split((qk_width, qk_width, d_model), dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.qk_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.qk_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.qk_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--batch-size", type=int, default=512)
=======
    p.add_argument("--batch-size", type=int, default=256)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE