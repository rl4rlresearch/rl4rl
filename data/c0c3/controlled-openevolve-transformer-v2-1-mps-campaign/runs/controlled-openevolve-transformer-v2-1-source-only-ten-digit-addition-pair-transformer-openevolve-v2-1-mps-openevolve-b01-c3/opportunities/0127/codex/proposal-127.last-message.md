MECHANISM: Six-unit MLP with final-pair secondary-head gain sharing

HYPOTHESIS: A 975-parameter transformer will maintain at least 99% accuracy because the six-unit, 976-parameter design achieved 100%, while the same isolated final-pair gain tie previously retained 99.88%; preserving six MLP units avoids the five-unit bottleneck that fell to 96.48%.

INTENDED_EDIT: Adopt the verified six-unit MLP and parameterize the seven secondary-head gains as `[a,b,c,d,e,f,f]`, removing one additional learned parameter.

EVIDENCE: The six-unit reference reached 100% at 976 parameters, whereas five units failed at 96.48%; independently, tying only the final secondary-head gain pair met the requirement at 99.88%, motivating that compression while retaining the proven MLP width.

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain,
            ),
            dim=0,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain,
                self.secondary_value_gain[..., -1:],
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
    p.add_argument("--d-ff", type=int, default=9)
=======
    p.add_argument("--d-ff", type=int, default=6)
>>>>>>> REPLACE