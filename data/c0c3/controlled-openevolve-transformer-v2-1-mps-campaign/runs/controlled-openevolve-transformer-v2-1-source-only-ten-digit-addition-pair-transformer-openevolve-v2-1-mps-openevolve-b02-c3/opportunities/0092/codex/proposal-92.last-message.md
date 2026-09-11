MECHANISM: Complete classifier-offset fixing

HYPOTHESIS: A 801-parameter transformer will retain at least 99% accuracy because fixing the final classifier-offset coefficient at zero preserves the full-rank tied token embedding, attention routing, all four gated nonlinear features, and all learned final normalization scales.

INTENDED_EDIT: Reconstruct the qualified bias-free gated MLP and remove the sole classifier-offset coefficient remaining in the 802-parameter design.

EVIDENCE: The 802-parameter design achieved 99.92% after reducing the classifier offset to one learned coefficient; removing that last coefficient is the narrowest reduction from the smallest qualified design, while the failed 689-parameter experiment warns against reducing lexical rank.

<<<<<<< SEARCH
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )

        self.apply(self._init_weights)
=======
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        final_bias = F.pad(self.final_bias, (0, 2))
        x = x + final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
=======
        x = self.ln_f(x)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE