MECHANISM: Two-pair secondary-head value gain

HYPOTHESIS: A 1,056-parameter transformer will maintain at least 99% accuracy because the verified 1,057-parameter model achieved 99.88%, and tying one additional adjacent pair preserves five independently learned secondary-head gains rather than collapsing all feature specialization.

INTENDED_EDIT: Preserve the verified final gain pair and tie the two preceding secondary-head gain coordinates, changing the expansion from `[a,b,c,d,e,f,f]` to `[a,b,c,d,d,e,e]`.

EVIDENCE: The current isolated final-pair tie retained 99.88%, while collapsing all seven gains to one scalar reached only 97.11%; additionally, the attention-bias experiments showed that preserving a successful group boundary while tying other coordinates can succeed when extending that group fails.

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 1)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :-1],
                self.secondary_value_gain[..., -1:].expand(-1, 2),
            ),
            dim=-1,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :-2],
                self.secondary_value_gain[..., -2:-1].expand(-1, 2),
                self.secondary_value_gain[..., -1:].expand(-1, 2),
            ),
            dim=-1,
        )
>>>>>>> REPLACE