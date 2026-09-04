MECHANISM: Single fixed classifier-bias direction

HYPOTHESIS: A 803-parameter transformer will retain at least 99% accuracy because fixing one final classifier-offset coefficient preserves the full-rank tied token embedding, all attention routing, and all four gated nonlinear features.

INTENDED_EDIT: Learn two rather than three final hidden-space bias coefficients, fixing the third coefficient at zero.

EVIDENCE: The 804-parameter design achieved 99.96% accuracy after all gated-MLP intercepts were removed; its margin supports the narrowest one-parameter reduction, while preserving lexical rank because the prior lexical-rank reduction reached only 95.12%.

<<<<<<< SEARCH
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 2)
        )
=======
        self.final_bias = nn.Parameter(
            torch.zeros(self.token_emb.code.embedding_dim - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        final_bias = F.pad(self.final_bias, (0, 2))
=======
        final_bias = F.pad(self.final_bias, (0, 3))
>>>>>>> REPLACE