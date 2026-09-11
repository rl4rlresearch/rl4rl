MECHANISM: Rank-two learned residual over a full-rank sinusoidal positional basis

HYPOTHESIS: Adding independent per-position corrections on residual coordinates 2 and 5 to the 97.63%-accurate sinusoidal design will raise accuracy to at least 99% while reducing the current model from 1,303 to 1,229 parameters.

INTENDED_EDIT: Replace the 23×8 learned positional table with a normalized fixed sinusoidal basis, a learned 8×8 projection, and a learned 23×2 positional residual injected into coordinates 2 and 5.

EVIDENCE: The full-rank sinusoidal design reached 97.63% with only 64 positional parameters, showing that structured positions nearly suffice; coordinates 2 and 5 are also the final-normalization scale coordinates that could not safely be fixed, motivating a small unstructured correction on those channels.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        position = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        pos_basis = torch.empty(cfg.max_seq_len, cfg.d_model)
        pos_basis[:, 0::2] = torch.sin(position * frequency)
        pos_basis[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer(
            "pos_basis",
            F.normalize(pos_basis, dim=-1),
            persistent=False,
        )
        self.pos_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_residual = nn.Embedding(cfg.max_seq_len, 2)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        fixed_pos = self.pos_proj(self.pos_basis[:seqlen]).unsqueeze(0)
        residual = self.pos_residual(pos)
        residual = F.pad(residual[..., :1], (2, 5)) + F.pad(
            residual[..., 1:], (5, 2)
        )
        x = self.token_emb(idx) + fixed_pos + residual
        x = self.drop(x)
>>>>>>> REPLACE