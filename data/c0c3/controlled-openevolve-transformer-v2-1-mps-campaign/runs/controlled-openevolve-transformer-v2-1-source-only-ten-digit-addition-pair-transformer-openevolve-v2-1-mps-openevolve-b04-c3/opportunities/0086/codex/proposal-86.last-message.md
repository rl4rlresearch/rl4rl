MECHANISM: Three-scalar tied-token residual-basis constraint

HYPOTHESIS: Fixing a third scalar of the tied six-dimensional token embedding will retain at least 99% accuracy after 45,000 updates while reducing the verified 1,144-parameter design to 1,143 learned parameters.

INTENDED_EDIT: Adopt the verified two-scalar token constraint and fix one additional trailing embedding scalar at zero, preserving full embedding initialization RNG consumption and tied input/output use.

EVIDENCE: Reference Design 1 achieved 100% accuracy with two fixed token-embedding scalars and 1,144 parameters, after the one-scalar version also achieved 100%; an incremental third constraint is the most direct evidence-backed reduction.

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Generic Fourier coordinates are synthesized by a shared dense map.
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        # Choose a basis for the tied token subspace by fixing its final three
        # scalars. Constructing the full Embedding above preserves constructor RNG.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 3)
        )
        self.token_emb.fixed_weight_trim = (0, 3)

        # Generic Fourier coordinates are synthesized by a shared dense map.
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
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
=======
        token_weight = F.pad(self.token_emb.weight, (0, 3)).view(
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