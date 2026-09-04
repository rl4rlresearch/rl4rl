MECHANISM: Head-specific learned relative-distance attention

HYPOTHESIS: Replacing the 161-parameter absolute positional table with 46 learned causal distance logits will produce a 1,449-parameter transformer with at least 99% accuracy, because addition’s reusable token alignments can be represented directly by attention offsets rather than reconstructed from absolute position vectors.

INTENDED_EDIT: Remove absolute positional embeddings and give each attention head a learned bias for every causal relative distance, while preserving the existing model’s initialization RNG stream for a clean comparison.

EVIDENCE: Compressing absolute positional embeddings to four dimensions reached only 73.99%, showing that positional information is load-bearing; the successful seven-contrast design retained 99.82%. This patch challenges the shared assumption that those relationships require residual-stream position vectors by moving positional representation into a fully learned, head-specific attention mechanism.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        distance = positions[:, None] - positions[None, :]
        att = att + self.rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model - 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
=======
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Preserve the constructor RNG stream of the removed position table.
        torch.empty(cfg.max_seq_len, cfg.d_model - 1).normal_()
        self.drop = nn.Dropout(cfg.dropout)
        self.drop._removed_pos_shape = (cfg.max_seq_len, cfg.d_model - 1)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        removed_pos_shape = getattr(module, "_removed_pos_shape", None)
        if removed_pos_shape is not None:
            torch.empty(removed_pos_shape).normal_()

        if isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_emb = F.pad(self.pos_emb(pos), (0, 1))
        x = self.token_emb(idx) + pos_emb
        x = self.drop(x)
=======
        x = self.token_emb(idx)
        x = self.drop(x)
>>>>>>> REPLACE