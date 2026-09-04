MECHANISM: Cross-head sharing of the five sparsest relative-distance biases

HYPOTHESIS: A 1,281-parameter model will retain at least 99% accuracy because the verified 1,282-parameter model achieved 99.94%, while additionally sharing the fifth-farthest bias affects only five query-key pairs at full context and preserves independent head-specific biases at every more frequent distance.

INTENDED_EDIT: Replace absolute position embeddings with gauge-anchored relative-attention biases, sharing the five maximum-distance biases across heads while retaining the current learned projections, MLP bias quartets, and attention-output bias tie.

EVIDENCE: The 1,282-parameter design met the requirement after progressively extending successful cross-head sharing from one through four sparsest distance bins; sharing the next-sparsest bin is the smallest directly supported reduction.

<<<<<<< SEARCH
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
=======
        self.proj = PairTiedBiasMeanZeroLinear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 6))
        self.far_rel_bias = nn.Parameter(torch.zeros(5))
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        relative_distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        learned_bias = torch.cat(
            (
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 5),
            ),
            dim=1,
        )
        learned_bias = F.pad(learned_bias, (1, 0))
        att = att + learned_bias[:, relative_distance].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = MeanZeroEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = VocabCenteredEmbedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.token_emb(idx)
        x = self.drop(x)
>>>>>>> REPLACE