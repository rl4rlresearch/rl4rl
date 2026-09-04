MECHANISM: Two-scalar tied-token residual-basis gauge fixing

HYPOTHESIS: The verified 1,145-parameter model will retain at least 99% accuracy after 45,000 updates when a second coordinate of the tied six-dimensional token code is fixed at zero, yielding 1,144 learned parameters.

INTENDED_EDIT: Learn all but two scalars of the tied token embedding, reconstructing both fixed trailing scalars as zeros while preserving full embedding initialization RNG consumption.

EVIDENCE: Fixing one tied-token scalar achieved 100% accuracy with 1,145 parameters, whereas further positional-mixer and attention-output-bias constraints failed; the smallest informative next probe is one additional constraint in the successful token-code mechanism.

<<<<<<< SEARCH
        # Choose a basis for the tied token subspace by fixing its final scalar.
        # The full Embedding constructor above preserves constructor RNG use.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 1)
        )
        self.token_emb.fixed_weight_trim = (0, 1)
=======
        # Choose a basis for the tied token subspace by fixing its final two scalars.
        # The full Embedding constructor above preserves constructor RNG use.
        self.token_emb.weight = nn.Parameter(
            torch.empty(cfg.vocab_size * self.token_dim - 2)
        )
        self.token_emb.fixed_weight_trim = (0, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_weight = F.pad(self.token_emb.weight, (0, 1)).view(
=======
        token_weight = F.pad(self.token_emb.weight, (0, 2)).view(
>>>>>>> REPLACE