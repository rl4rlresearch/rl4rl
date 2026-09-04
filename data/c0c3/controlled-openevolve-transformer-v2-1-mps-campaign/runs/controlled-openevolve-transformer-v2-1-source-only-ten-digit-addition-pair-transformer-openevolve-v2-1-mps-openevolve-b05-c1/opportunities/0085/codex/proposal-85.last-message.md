MECHANISM: Second function-preserving query/key rotation in the first attention head

HYPOTHESIS: Fixing a second centered query/key coefficient in the first head will reduce the model from 1,270 to 1,269 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate the first head’s remaining center-0/center-2 plane at initialization, omit center-2’s first coefficient, and reconstruct it as zero during forward passes.

EVIDENCE: One independently initialized rotational gauge per head retained 99.99% accuracy at 1,270 parameters, while imposing all six gauges at once failed; adding one constraint to a single head is the smallest supported continuation.

<<<<<<< SEARCH
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2 = nn.Parameter(torch.empty(1, in_features))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
=======
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2.copy_(coeff[3:4])

            dense_second = torch.empty_like(
=======
            first = coeff[1].clone()
            second = coeff[3].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
            coeff[1] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2_tail.copy_(coeff[3:4, 1:])

            dense_second = torch.empty_like(
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qk_center0,
                F.pad(self.qk_center1_tail, (1, 0)),
                self.qk_center2,
            ),
=======
                self.qk_center0,
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (1, 0)),
            ),
>>>>>>> REPLACE