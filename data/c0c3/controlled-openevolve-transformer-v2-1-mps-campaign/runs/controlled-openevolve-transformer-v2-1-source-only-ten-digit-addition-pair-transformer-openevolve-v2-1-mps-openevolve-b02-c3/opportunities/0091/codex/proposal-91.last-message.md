MECHANISM: Second fixed classifier-bias direction

HYPOTHESIS: A 802-parameter transformer will retain at least 99% accuracy because fixing a second final classifier-offset coefficient preserves the full-rank tied token embedding, all attention routing, and all four gated nonlinear features.

INTENDED_EDIT: Reconstruct the qualified 803-parameter bias-free gated MLP and additionally learn only one final hidden-space bias coefficient, fixing the remaining four coefficients at zero.

EVIDENCE: The 803-parameter design achieved 99.81% accuracy with two learned final-bias coefficients; removing one more coefficient is the narrowest reduction from the smallest qualified design, while the failed 689-parameter result warns against reducing lexical rank.

<<<<<<< SEARCH
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff)
=======
        self.fc1 = QuotientInputLinear(d_model, 2 * d_ff, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        final_bias = F.pad(self.final_bias, (0, 2))
=======
        final_bias = F.pad(self.final_bias, (0, 4))
>>>>>>> REPLACE