MECHANISM: Final LayerNorm null-direction quotient bias

HYPOTHESIS: Restricting the final LayerNorm bias to the fixed seven-dimensional basis will reduce parameters from 1,460 to 1,459 while retaining at least 99% accuracy, because the rank-seven lexical projection cannot observe its remaining bias direction.

INTENDED_EDIT: Make the final LayerNorm bias-free and add a seven-parameter post-normalization bias expressed in the lexical basis.

EVIDENCE: The current gauge-fixed rank-seven model achieved 99.96% accuracy. Unlike the failed removal of `ln2` bias, this preserves every output-relevant bias degree of freedom because the lexical projection satisfies `WᵀB = I`.

<<<<<<< SEARCH
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.final_bias = nn.Parameter(torch.zeros(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.token_emb.logits(x)
=======
        x = self.ln_f(x)
        x = x + self.final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE