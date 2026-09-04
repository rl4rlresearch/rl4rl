MECHANISM: Rank-six tied vocabulary embedding with a parameter-free channel lift

HYPOTHESIS: Retaining the current learned positional table while constraining the tied token embedding to a six-dimensional learned subspace will achieve at least 99% accuracy after 45,000 updates with 1,360 learned parameters.

INTENDED_EDIT: Replace the 8-dimensional tied token matrix with six learned coordinates padded into the 8-dimensional residual stream, preserve exact input/output weight tying, and validate only at the final positive step.

EVIDENCE: The current learned-position model reached 99.92% in 45,000 steps, whereas multiple smaller sinusoidal-position models failed at 45,000 steps or timed out at longer schedules; this motivates preserving the proven positional and optimization backbone while removing 228 vocabulary-embedding parameters.

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.cfg = cfg

        # Token identities use a learned rank-six code. A fixed zero lift places
        # that code in the full residual stream without adding parameters; the
        # same lifted matrix is used by the output classifier below.
        self.token_dim = cfg.d_model - 2
        if self.token_dim <= 0:
            raise ValueError("d_model must be at least 3")
        self.token_scale = math.sqrt(cfg.d_model / self.token_dim)
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        x = self.token_scale * token_x + self.pos_emb(pos)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(
            x[..., : self.token_dim],
            self.token_scale * self.token_emb.weight,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE