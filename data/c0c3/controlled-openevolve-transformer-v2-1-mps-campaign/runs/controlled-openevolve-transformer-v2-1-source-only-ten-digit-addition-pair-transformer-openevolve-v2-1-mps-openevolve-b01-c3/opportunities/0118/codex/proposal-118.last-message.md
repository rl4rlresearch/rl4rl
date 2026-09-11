MECHANISM: Three-pair secondary-head value-gain sharing

HYPOTHESIS: A 1,055-parameter transformer will maintain at least 99% accuracy because the verified 1,056-parameter `[a,b,c,d,d,e,e]` gain design achieved 100%, and tying the next adjacent pair retains four independently learned feature gains.

INTENDED_EDIT: Change the secondary-head value-gain expansion to `[a,b,b,c,c,d,d]`, removing one learned parameter while preserving all attention routing, biases, projections, and residual paths.

EVIDENCE: The 1,056-parameter two-pair gain design achieved 100%, while collapsing all seven gains to one scalar failed at 97.11%; this motivates another isolated adjacent-pair tie rather than broad gain sharing.

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain,
            ),
            dim=0,
        )
=======
        y = att @ v
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:2].expand(-1, 2),
                self.secondary_value_gain[..., 2:3].expand(-1, 2),
                self.secondary_value_gain[..., 3:].expand(-1, 2),
            ),
            dim=-1,
        )
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                secondary_value_gain,
            ),
            dim=0,
        )
>>>>>>> REPLACE