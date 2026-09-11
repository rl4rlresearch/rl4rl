MECHANISM: Symmetric next-longest-distance attention-bias fixing

HYPOTHESIS: A 799-parameter transformer will retain at least 99% accuracy because fixing both heads’ sparsely used next-longest relative-position bias preserves the qualified model’s full-rank lexical path, normalization scales, attention projections, and nonlinear features.

INTENDED_EDIT: Reconstruct the qualified 800-parameter design, then fix the first head’s next-longest relative-attention bias in addition to the already qualified fixed coefficient for the second head.

EVIDENCE: The 800-parameter reference achieved 99.82% after fixing one head’s next-longest relative-position coefficient; applying the same constraint to the corresponding equally sparse coefficient of the other head is the narrowest adjacent reduction, while prior reductions to final normalization or Q/K projections failed sharply.

<<<<<<< SEARCH
        # Distance zero cancels inside softmax, and the uniquely sparse
        # maximum distance is fixed at zero for both heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
=======
        # Distance zero cancels inside softmax, while the two sparsest
        # maximum distances are fixed at zero for both heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
=======
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 2),
            ),
            dim=1,
        )
>>>>>>> REPLACE

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