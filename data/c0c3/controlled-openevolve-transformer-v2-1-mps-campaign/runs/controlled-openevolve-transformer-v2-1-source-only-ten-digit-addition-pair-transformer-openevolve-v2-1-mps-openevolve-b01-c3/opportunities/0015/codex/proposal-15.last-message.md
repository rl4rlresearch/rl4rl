MECHANISM: Headwise query-bias rotational gauge fixing

HYPOTHESIS: Fixing one query-bias coordinate per attention head will reduce the model from 1,534 to 1,532 parameters while retaining at least 99% accuracy, because a headwise orthogonal rotation can move any query-bias vector into the remaining three coordinates while applying the same rotation to queries and keys leaves every attention logit unchanged.

INTENDED_EDIT: Store three learned query-bias coordinates per four-dimensional head, reconstructing the fourth as zero, while preserving full query/key projection widths and the verified terminal normalization.

EVIDENCE: The 1,534-parameter design achieved 100% accuracy, whereas narrowing query/key width reached only 98.18% and adding a second terminal-scale anchor collapsed to 72%; this motivates preserving those capacities and removing only an exact headwise attention-coordinate redundancy.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head, self.head_dim - 1))
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(self.q_bias, (0, 1)).reshape(d_model)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE