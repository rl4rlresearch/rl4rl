MECHANISM: Adjacent extreme-distance attention-bias tying

HYPOTHESIS: A 1,269-parameter model will retain at least 99% accuracy because tying the final two already head-shared relative-bias bins removes one parameter while constraining only the single farthest query-key pair per head beyond the penultimate bin.

INTENDED_EDIT: Replace the fourteen independent far-distance scalars with thirteen scalars, reusing the final scalar for both extreme-distance bins.

EVIDENCE: The current fourteen-bin tilted-bias model achieved 99.31%; unlike failed 1,269-parameter changes that altered load-bearing projections, normalization, or cross-module optimization, this preserves the model everywhere except its sparsest relative-distance distinction.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
        self.attn_drop = nn.Dropout(dropout)
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(13))
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        learned_bias = torch.cat(
            (
                self.rel_bias,
                self.far_rel_bias.expand(self.n_head, 14),
            ),
            dim=1,
        )
=======
        far_rel_bias = torch.cat(
            (self.far_rel_bias, self.far_rel_bias[-1:])
        )
        learned_bias = torch.cat(
            (
                self.rel_bias,
                far_rel_bias.expand(self.n_head, 14),
            ),
            dim=1,
        )
>>>>>>> REPLACE