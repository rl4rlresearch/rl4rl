MECHANISM: Initialization-scaled diagonal query–key gauge fixing

HYPOTHESIS: Extending the qualified 1490-parameter model by fixing first-head query row 3, input coordinate 0 to its initialization-scale norm and inversely scaling key row 3 will yield 1489 parameters, at least 99% accuracy, and finish within the verification limit.

INTENDED_EDIT: Reproduce the qualified three query shears, then remove the remaining query-row-3 anchor parameter through an exact, well-conditioned diagonal query–key rescaling.

EVIDENCE: The three-shear 1490-parameter design achieved 99.92% accuracy; subsequent 1489 attempts timed out only after adding a fourth shear. The unused diagonal GL symmetry uses the robust norm-bearing row-3 coordinate and adds no training-time transformation.

<<<<<<< SEARCH
        self.second_query = head_dim
        basis = _mean_zero_basis(in_features)
=======
        self.second_query = head_dim
        self.query_anchor = 0.02 * math.sqrt(head_dim)
        basis = _mean_zero_basis(in_features)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.third_weight = nn.Parameter(fixed_weight[2, 1:])
        self.first_head_tail = nn.Parameter(
            fixed_weight[3:self.second_query]
        )
        self.head_two_weight = nn.Parameter(
=======
        self.third_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[2, 1:2],
                    fixed_weight[2, 3:],
                )
            )
        )
        self.fourth_weight = nn.Parameter(fixed_weight[3, 3:])
        self.head_two_weight = nn.Parameter(
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[query_start, input_coord] = 0.0

        value_rotations = []
=======
            fixed_weight[query_start, input_coord] = 0.0

        key_start = self.in_features
        shear = -fixed_weight[2, 2] / fixed_weight[1, 2]
        fixed_weight[2] = fixed_weight[2] + shear * fixed_weight[1]
        fixed_weight[key_start + 1] = (
            fixed_weight[key_start + 1]
            - shear * fixed_weight[key_start + 2]
        )
        fixed_weight[2, 2] = 0.0

        second_shear = -fixed_weight[3, 1] / fixed_weight[2, 1]
        fixed_weight[3] = (
            fixed_weight[3] + second_shear * fixed_weight[2]
        )
        fixed_weight[key_start + 2] = (
            fixed_weight[key_start + 2]
            - second_shear * fixed_weight[key_start + 3]
        )
        fixed_weight[3, 1] = 0.0

        third_shear = -fixed_weight[3, 2] / fixed_weight[1, 2]
        fixed_weight[3] = (
            fixed_weight[3] + third_shear * fixed_weight[1]
        )
        fixed_weight[key_start + 1] = (
            fixed_weight[key_start + 1]
            - third_shear * fixed_weight[key_start + 3]
        )
        fixed_weight[3, 2] = 0.0

        diagonal_scale = self.query_anchor / fixed_weight[3, 0]
        fixed_weight[3] = diagonal_scale * fixed_weight[3]
        fixed_weight[key_start + 3] = (
            fixed_weight[key_start + 3] / diagonal_scale
        )
        fixed_weight[3, 0] = self.query_anchor

        value_rotations = []
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.third_weight.copy_(fixed_weight[2, 1:])
            self.first_head_tail.copy_(
                fixed_weight[3:self.second_query]
            )
            self.head_two_weight.copy_(
=======
            self.third_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[2, 1:2],
                        fixed_weight[2, 3:],
                    )
                )
            )
            self.fourth_weight.copy_(fixed_weight[3, 3:])
            self.head_two_weight.copy_(
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_row = F.pad(self.third_weight, (1, 0))
        head_two_row = F.pad(self.head_two_weight, (3, 0))
=======
        third_row = torch.cat(
            (
                self.third_weight.new_zeros(1),
                self.third_weight[:1],
                self.third_weight.new_zeros(1),
                self.third_weight[1:],
            )
        )
        fourth_row = torch.cat(
            (
                self.fourth_weight.new_full((1,), self.query_anchor),
                self.fourth_weight.new_zeros(2),
                self.fourth_weight,
            )
        )
        head_two_row = F.pad(self.head_two_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
                third_row.unsqueeze(0),
                self.first_head_tail,
                head_two_row.unsqueeze(0),
=======
                third_row.unsqueeze(0),
                fourth_row.unsqueeze(0),
                head_two_row.unsqueeze(0),
>>>>>>> REPLACE