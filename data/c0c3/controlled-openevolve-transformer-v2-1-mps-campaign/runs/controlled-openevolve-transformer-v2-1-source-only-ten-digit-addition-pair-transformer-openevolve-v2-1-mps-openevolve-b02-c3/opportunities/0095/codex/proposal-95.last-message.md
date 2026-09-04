MECHANISM: Asymmetric longest-learned-distance bias fixing

HYPOTHESIS: A 800-parameter transformer will retain at least 99% accuracy because fixing one head’s sparsely used longest learned relative-position bias preserves the other head’s coefficient, while the qualified 801-parameter model’s full-rank lexical path, attention projections, MLP features, and normalization scales remain intact.

INTENDED_EDIT: Reconstruct the qualified 801-parameter design by removing the classifier offset, then fix only the last head’s longest currently learned relative-attention bias at zero.

EVIDENCE: The 801-parameter design achieved 99.51% accuracy; unlike the failed 800-parameter changes to final normalization and Q/K projections, this removes one coefficient from the next-sparsest relative distance, adjacent to the already-fixed uniquely sparse maximum distance.

<<<<<<< SEARCH
        # Distance zero cancels inside softmax, and the uniquely sparse
        # maximum distance is fixed at zero for both heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 2))
=======
        # Distance zero and the uniquely sparse maximum distance are fixed.
        # The last head's next-longest distance is also fixed at zero.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 2) - 1)
        )
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
        relative_bias_coeff = F.pad(self.relative_bias, (0, 1)).view(
            self.n_head, -1
        )
        learned_relative_bias = torch.cat(
            (
                relative_bias_coeff,
                relative_bias_coeff.new_zeros(self.n_head, 1),
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = ClassifierAwareLayerNorm(
            cfg.d_model, self.token_emb.code.embedding_dim + 1
        )
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 3)
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
        final_bias = F.pad(self.final_bias, (0, 3))
        x = x + final_bias @ self.token_emb.basis.transpose(0, 1)
        logits = self.token_emb.logits(x)
=======
        x = self.ln_f(x)
        logits = self.token_emb.logits(x)
>>>>>>> REPLACE