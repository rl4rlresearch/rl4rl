MECHANISM: Learned relative-lag attention routing

HYPOTHESIS: Replacing full-width absolute position embeddings with per-head learned relative-lag logits will retain at least 99% accuracy while removing `6 * max_seq_len - 2` parameters, because addition dependencies repeat at relative offsets and content-dependent query/key attention remains intact.

INTENDED_EDIT: Remove absolute positions from the residual stream and let each causal attention head learn its own lag preference directly; simplify gauge optimization accordingly. This challenges the prior assumption that every position needs a full learned vector.

EVIDENCE: Absolute-position compression stalled at 97.06% for a rank-4 subspace and 94.19% for fixed Chebyshev features, whereas the shared-key/value attention design reached 99.98%. This motivates changing positional computation itself instead of further compressing the same absolute-table representation.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        # Each head learns which causal lags are useful, providing positional
        # routing without injecting a full vector for every absolute position.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len)
        )
        self.proj = GaugeFixedAttentionProjection(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        lags = (
            positions.unsqueeze(1) - positions.unsqueeze(0)
        ).clamp_min(0)
        att = att + self.relative_bias[:, lags].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedPositionEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # two positional, three attention-weight, attention-bias, terminal-bias,
    # and six terminal-weight gauges.
    gauge_params = [
        model.token_emb.weight,
        model.pos_emb.first,
        model.pos_emb.second,
    ]
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # three attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
    gauge_params = [
        model.token_emb.weight,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
            model.pos_emb.full_first.grad.detach(),
            model.pos_emb.full_second.grad.detach(),
        ]
=======
        full_gauge_grads = [
            model.token_emb.full_weight.grad.detach().reshape(-1),
        ]
>>>>>>> REPLACE