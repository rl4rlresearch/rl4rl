MECHANISM: Tied-embedding grand-common-mode quotient

HYPOTHESIS: Removing the single grand-common-mode direction from the tied token embedding will yield 1,617 parameters and at least 99% accuracy, because it changes every input embedding by the same uniform residual-channel shift and every output logit by the same scalar, leaving the modeled distribution unchanged.

INTENDED_EDIT: Represent the tied token embedding in a one-dimension-smaller orthonormal zero-sum basis, reconstruct it for input lookup and output logits, and train it with the existing full-coordinate AdamW quotient updates.

EVIDENCE: The current 1,618-parameter design achieved 99.94%, while larger `fc2.weight` and additional `ln2.bias` quotients failed; this tests one independent, exactly function-null direction without further constraining those sensitive components.

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
=======
        self.apply(self._init_weights)

        # Adding one scalar to every coordinate of the tied embedding shifts
        # every input residual uniformly across channels, which all LayerNorms
        # erase, and adds only a vocabulary-uniform output-logit shift.
        token_size = cfg.vocab_size * cfg.d_model
        token_basis = torch.zeros(token_size, token_size - 1)
        for col in range(token_size - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            token_basis[: col + 1, col] = 1.0 / scale
            token_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            compact_token = (
                self.token_emb.weight.detach().reshape(-1) @ token_basis
            )
        self.token_emb.weight = nn.Parameter(compact_token)
        self.lm_head.weight = self.token_emb.weight
        self.register_buffer("token_basis", token_basis, persistent=False)

        # Preserve baseline initialization and RNG consumption, then remove
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        compact_size = self.compact_pos_count * (self.cfg.d_model - 1)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_weight = (
            self.token_basis @ self.token_emb.weight
        ).view(self.cfg.vocab_size, self.cfg.d_model)
        compact_size = self.compact_pos_count * (self.cfg.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
=======
        x = F.embedding(idx, token_weight) + F.embedding(pos, pos_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        logits = self.lm_head(x)
=======
        logits = F.linear(x, token_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    ]
    pos_param = model.pos_emb.weight
=======
    ]
    gauge_params.append((model.token_emb.weight, model.token_basis))
    pos_param = model.pos_emb.weight
>>>>>>> REPLACE