MECHANISM: Within-head query–key shear gauge fixing

HYPOTHESIS: Applying one inverse-compensated first-head query–key shear will reduce the qualified model from 1493 to 1492 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: After the twelve orthogonal query rotations, add a triangular shear that fixes first-head query row 2, input coordinate 2 at zero, applies the inverse-transpose shear to the corresponding key rows, and reconstructs the removed query coordinate during every forward pass.

EVIDENCE: The twelfth query–key rotation preserved 99.97% accuracy at 1494 parameters, and the subsequent value–output rotation preserved 99.97% at 1493; the failed 1492 attempt recanonicalized the sensitive first value head, so testing an unused exact query–key GL symmetry is the most informative independent reduction.

<<<<<<< SEARCH
        self.third_weight = nn.Parameter(
            fixed_weight[2, 1:]
        )
=======
        self.third_weight = nn.Parameter(
            torch.cat(
                (
                    fixed_weight[2, 1:2],
                    fixed_weight[2, 3:],
                )
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            fixed_weight[query_start, input_coord] = 0.0

        value_rotations = []
=======
            fixed_weight[query_start, input_coord] = 0.0

        # Use the remaining triangular GL gauge after orthogonal
        # canonicalization.  Q' = A Q and K' = A^{-T} K preserve QK^T.
        shear = -fixed_weight[2, 2] / fixed_weight[1, 2]
        fixed_weight[2] = fixed_weight[2] + shear * fixed_weight[1]
        key_start = self.in_features
        fixed_weight[key_start + 1] = (
            fixed_weight[key_start + 1]
            - shear * fixed_weight[key_start + 2]
        )
        fixed_weight[2, 2] = 0.0

        value_rotations = []
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.third_weight.copy_(fixed_weight[2, 1:])
=======
            self.third_weight.copy_(
                torch.cat(
                    (
                        fixed_weight[2, 1:2],
                        fixed_weight[2, 3:],
                    )
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        third_row = F.pad(self.third_weight, (1, 0))
=======
        third_row = torch.cat(
            (
                self.third_weight.new_zeros(1),
                self.third_weight[:1],
                self.third_weight.new_zeros(1),
                self.third_weight[1:],
            )
        )
>>>>>>> REPLACE