MECHANISM: Sinusoidal absolute positions with learned relative-lag attention

HYPOTHESIS: A full-rank sinusoidal positional basis plus per-head learned causal-distance biases will reach at least 99% accuracy with 1,226 parameters, because attention can learn operand alignments directly instead of reconstructing them from independently learned absolute vectors.

INTENDED_EDIT: Replace the 154-parameter independent positional table with a learned 8×8 projection of fixed anchored sinusoidal codes and add 44 gauge-fixed per-head relative-attention biases.

EVIDENCE: The sinusoidal projection reached 97.63%, showing structured absolute positions retain nearly all required information, while a learned residual correction failed at 30.60%. Injecting the missing positional flexibility directly into attention logits tests a different mechanism tailored to pairwise alignment.

<<<<<<< SEARCH
        self.qkv = MeanZeroInputLinear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.qkv = MeanZeroInputLinear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        # Distance zero is the fixed reference because softmax eliminates a
        # head-wide common shift of all relative-position logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        distance = (positions[:, None] - positions[None, :]).clamp_min(0)
        relative_bias = F.pad(self.relative_bias, (1, 0))
        att = att + relative_bias[:, distance].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len - 1, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.pos_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        positions = torch.arange(
            cfg.max_seq_len, dtype=torch.float32
        ).unsqueeze(1)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            / cfg.d_model
        )
        pos_basis = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_basis[:, 0::2] = torch.sin(positions * frequencies)
        pos_basis[:, 1::2] = torch.cos(positions * frequencies)
        pos_basis = pos_basis - pos_basis[:1]
        pos_basis = F.normalize(pos_basis, p=2.0, dim=-1)
        self.register_buffer("pos_basis", pos_basis, persistent=False)

        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = F.pad(self.pos_emb.weight, (0, 0, 1, 0))
        x = F.embedding(idx, self.token_weight()) + F.pad(
            F.embedding(pos, pos_weight), (0, 1)
        )
=======
        x = F.embedding(idx, self.token_weight()) + self.pos_proj(
            self.pos_basis[:seqlen]
        ).unsqueeze(0)
>>>>>>> REPLACE