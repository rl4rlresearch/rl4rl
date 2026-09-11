MECHANISM: Low-rank absolute codes with head-specific relative-distance attention

HYPOTHESIS: Replacing full 8-dimensional positional embeddings with learned rank-4 codes while adding per-head relative-distance attention biases will reduce the model below 1,600 parameters and retain at least 99% accuracy, because addition primarily needs relational alignment between output and operand positions rather than eight independent absolute-position coordinates.

INTENDED_EDIT: Factorize the learned absolute positional table to rank 4 using an RNG-preserving truncated SVD initialization, and augment each causal attention head with a learned bias indexed by query-key distance.

EVIDENCE: Positional gauge results were sharply coordinate-sensitive—two removed scalars achieved 99.75%, while a third collapsed to 73.73%—and shared-key attention collapsed to 36.45%. This motivates preserving full query/key/value content projections while moving positional alignment into a direct learned relative-attention mechanism.

<<<<<<< SEARCH
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
=======
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Each head learns which causal distances are useful, allowing operand
        # alignment to be represented directly in attention-score space.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len))

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        att = att + self.relative_bias[:, distance].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Fix two exact token/position embedding gauges while preserving the
        # initialized function: input embeddings are unchanged, and the tied
        # output weights gain only token-independent logit shifts.
        with torch.no_grad():
            offset = self.pos_emb.weight[0, :2].clone()
            self.token_emb.weight[:, :2].add_(offset)
            shifted_pos = self.pos_emb.weight.clone()
            shifted_pos[:, :2].sub_(offset)
            compact_pos = torch.cat((shifted_pos[0, 2:], shifted_pos[1:].reshape(-1)))
        self.pos_emb.weight = nn.Parameter(compact_pos)
=======
        self.apply(self._init_weights)

        # Preserve the baseline RNG stream by initializing the original table,
        # then replace it with its best rank-four factorization.
        with torch.no_grad():
            left, singular, right = torch.linalg.svd(
                self.pos_emb.weight, full_matrices=False
            )
            scale = singular[:4].sqrt()
            pos_code = left[:, :4] * scale
            pos_proj = scale.unsqueeze(1) * right[:4]
        self.pos_code = nn.Parameter(pos_code)
        self.pos_proj = nn.Parameter(pos_proj)
        self.pos_emb = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = torch.cat(
            (self.pos_emb.weight.new_zeros(2), self.pos_emb.weight)
        ).view(self.cfg.max_seq_len, self.cfg.d_model)
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        position = F.embedding(pos, self.pos_code) @ self.pos_proj
        x = self.token_emb(idx) + position
        x = self.drop(x)
>>>>>>> REPLACE