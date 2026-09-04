MECHANISM: Asymmetric third-longest-distance attention-bias fixing

HYPOTHESIS: A 798-parameter transformer will retain at least 99% accuracy because fixing one head’s third-longest relative-position bias preserves the full-rank lexical path, normalization scales, attention projections, nonlinear features, and the other head’s coefficient at that distance.

INTENDED_EDIT: Flatten the learned relative-bias coefficients, remove the last head’s longest remaining learned coefficient, and reconstruct it as zero during the forward pass.

EVIDENCE: The 799-parameter design achieved 99.75% after fixing both heads’ two sparsest maximum-distance biases; its margin motivates the narrowest adjacent reduction at the next-sparsest distance while avoiding the final-normalization and Q/K constraints that failed sharply.

<<<<<<< SEARCH
        # Distance zero cancels inside softmax, while the two sparsest
        # maximum distances are fixed at zero for both heads.
        self.relative_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 3))
=======
        # Distance zero and the two sparsest maximum distances are fixed
        # for both heads; the last head's next-sparsest bias is also fixed.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head * (max_seq_len - 3) - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_relative_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 2),
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
                relative_bias_coeff.new_zeros(self.n_head, 2),
            ),
            dim=1,
        )
>>>>>>> REPLACE