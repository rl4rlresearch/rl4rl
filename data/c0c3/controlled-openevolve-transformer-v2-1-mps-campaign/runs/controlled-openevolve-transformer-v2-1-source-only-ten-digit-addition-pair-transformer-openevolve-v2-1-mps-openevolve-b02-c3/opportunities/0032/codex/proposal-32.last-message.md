MECHANISM: Four-coordinate tied lexical bottleneck

HYPOTHESIS: Reducing the qualified five-coordinate token embedding and classifier to four coordinates will produce a 934-parameter transformer with at least 99% accuracy, because the five-coordinate model achieved 99.93% and the preceding six-to-five reduction caused no meaningful accuracy loss.

INTENDED_EDIT: Apply the qualified quotient-input MLP and five-direction pre-MLP bias, then reduce the tied token/output code from five coordinates to four and align the final classifier bias with that subspace.

EVIDENCE: The 1,049-parameter five-coordinate design achieved 99.93%, slightly exceeding the 1,164-parameter six-coordinate design’s 99.92%; this supports testing another lexical-rank reduction while preserving the attention architecture whose direct-operator replacement failed at 89.02%.

<<<<<<< SEARCH
        if rank != embedding_dim - 1:
            raise ValueError("rank must equal embedding_dim - 1")
=======
        if rank < 1 or rank >= embedding_dim:
            raise ValueError("rank must be between one and embedding_dim - 1")
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = BiasOnlyLayerNorm(cfg.d_model)
=======
        self.ln2 = GaugeFixedLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 1
        )
=======
        self.token_emb = FactorizedTokenEmbedding(
            cfg.vocab_size, cfg.d_model, cfg.d_model - 4
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(torch.zeros(cfg.d_model - 1))
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim)
        )
>>>>>>> REPLACE