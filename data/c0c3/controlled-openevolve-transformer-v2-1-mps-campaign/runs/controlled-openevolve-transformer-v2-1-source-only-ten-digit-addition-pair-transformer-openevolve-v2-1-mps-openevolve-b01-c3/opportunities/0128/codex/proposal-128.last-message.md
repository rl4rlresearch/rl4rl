MECHANISM: Six-unit MLP with two-pair secondary-head gain sharing

HYPOTHESIS: The resulting 974-parameter transformer will maintain at least 99% accuracy because the verified six-unit, final-pair-tied 975-parameter model achieved 99.98%, while extending gain sharing from `[a,b,c,d,e,f,f]` to `[a,b,c,d,d,e,e]` independently achieved 100% in the larger-MLP design.

INTENDED_EDIT: Reduce the MLP width from ten to six units and represent the seven secondary-head gains with five learned values expanded as `[a,b,c,d,d,e,e]`.

EVIDENCE: The six-unit MLP is the narrowest verified bottleneck at 100%, and its 975-parameter final-pair-tied variant reached 99.98%; the independently verified two-pair gain pattern reached 100%, motivating one additional isolated gain tie without crossing the failed five-unit bottleneck.

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
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
                self.secondary_value_gain[..., :3],
                self.secondary_value_gain[..., 3:4].expand(-1, 2),
                self.secondary_value_gain[..., -1:].expand(-1, 2),
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

<<<<<<< SEARCH
    p.add_argument("--d-ff", type=int, default=10)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE