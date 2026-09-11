MECHANISM: Four-scalar tied-token residual-basis constraint

HYPOTHESIS: Fixing a fourth scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,143-parameter design to 1,142 learned parameters.

INTENDED_EDIT: Adopt the qualified two-coordinate attention-output-bias constraint and reconstruct the final four token-embedding scalars as zeros, preserving full initialization RNG consumption and tied input/output use.

EVIDENCE: The 1,143-parameter reference achieved 100% accuracy with three fixed token-embedding scalars, after the one- and two-scalar versions also achieved 100%; one additional token constraint is the smallest evidence-backed reduction.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        # The two token-free residual channels provide one rotational gauge.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        self.proj = nn.Linear(d_model, d_model)
        # The final two attention-output bias coordinates are fixed at zero.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = F.pad(self.proj.bias, (0, 1))
=======
        proj_bias = F.pad(self.proj.bias, (0, 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_scale = math.sqrt(cfg.d_model / self.token_dim)
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Generic Fourier coordinates are synthesized by the qualified dense
=======
        self.token_scale = math.sqrt(cfg.d_model / self.token_dim)
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        # Choose a basis for the tied token subspace by fixing its final four
        # scalars. Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 4)
        )
        self.token_emb.fixed_weight_trim = (0, 4)

        # Generic Fourier coordinates are synthesized by the qualified dense
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
=======
        elif isinstance(module, nn.Embedding):
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.num_embeddings * module.embedding_dim
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
            self.cfg.vocab_size,
            self.token_dim,
        )
        token_x = F.pad(
            F.embedding(idx, token_weight),
            (0, self.cfg.d_model - self.token_dim),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = F.linear(
            x[..., : self.token_dim],
            self.token_scale * self.token_emb.weight,
        )
=======
        logits = F.linear(
            x[..., : self.token_dim],
            self.token_scale * token_weight,
        )
>>>>>>> REPLACE