MECHANISM: Cross-coordinate query/key rotation gauge

HYPOTHESIS: Fixing the second head’s second rotationally redundant coefficient at input coordinate 1 will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy, because it preserves the dense initialization exactly while avoiding two gauge constraints on the same input coordinate.

INTENDED_EDIT: Rotate the second head’s center-0/center-2 plane using input coordinate 1, omit center-2’s coordinate-1 coefficient, and reconstruct it as zero during forward passes.

EVIDENCE: Two first-head rotations using coordinate 0 retained 100% accuracy at 1,269 parameters, but imposing the analogous second rotation on the second head fell to 73.74%. Changing only the second pivot tests whether that failure arose from the conditioning of concentrating both constraints on one input coordinate rather than from removing the exact rotational redundancy itself.

<<<<<<< SEARCH
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
=======
        self.qk_second_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
=======
            first = second_coeff[1].clone()
            second = second_coeff[3].clone()
            radius = torch.sqrt(first[1].square() + second[1].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[1] / radius
            sine = second[1] / radius
            second_coeff[1] = cosine * first + sine * second
            second_coeff[3] = -sine * first + cosine * second

            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2_tail.copy_(
                torch.cat((second_coeff[3:4, :1], second_coeff[3:4, 2:]), dim=1)
            )
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_head_coeff = torch.cat(
            (
                self.qk_second_common,
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                self.qk_second_center2,
            ),
            dim=0,
        )
=======
        second_center2 = torch.cat(
            (
                self.qk_second_center2_tail[:, :1],
                self.qk_second_center2_tail.new_zeros(1, 1),
                self.qk_second_center2_tail[:, 1:],
            ),
            dim=1,
        )
        second_head_coeff = torch.cat(
            (
                self.qk_second_common,
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                second_center2,
            ),
            dim=0,
        )
>>>>>>> REPLACE