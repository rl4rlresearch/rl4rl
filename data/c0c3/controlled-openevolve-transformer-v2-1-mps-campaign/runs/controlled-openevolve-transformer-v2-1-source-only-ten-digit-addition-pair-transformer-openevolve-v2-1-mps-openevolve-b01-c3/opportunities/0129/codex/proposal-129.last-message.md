MECHANISM: Six-unit MLP with three-pair secondary-head gain sharing

HYPOTHESIS: The resulting 973-parameter transformer will maintain at least 99% accuracy because the current 974-parameter model achieved 99.97%, while the same `[a,b,b,c,c,d,d]` gain pattern achieved 99.88% in the larger-MLP design.

INTENDED_EDIT: Replace the five learned secondary-head gains expanded as `[a,b,c,d,d,e,e]` with four gains expanded as `[a,b,b,c,c,d,d]`, removing one learned parameter while retaining the verified six-unit MLP.

EVIDENCE: The current six-unit, two-pair-gain model reached 99.97%, and the independently verified three-pair pattern reached 99.88%; this supports testing one isolated additional adjacent gain tie without crossing the failed five-unit MLP bottleneck.

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :3],
                self.secondary_value_gain[..., 3:4].expand(-1, 2),
                self.secondary_value_gain[..., -1:].expand(-1, 2),
            ),
            dim=-1,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
>>>>>>> REPLACE