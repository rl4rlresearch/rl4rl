MECHANISM: Second function-preserving query/key rotation in the second attention head

HYPOTHESIS: Applying the proven second centered rotation gauge to the second head will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate the second head’s remaining center-0/center-2 plane at initialization, omit center-2’s first coefficient, and reconstruct it as zero during forward passes.

EVIDENCE: The identical second rotation in the first head retained 100% accuracy at 1,269 parameters, while one rotation in each head retained 99.99%; the untouched second-head rotation is the closest supported one-parameter reduction.

<<<<<<< SEARCH
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
=======
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            second_coeff[1] = cosine * first + sine * second
            second_coeff[2] = -sine * first + cosine * second

            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])
=======
            second_coeff[1] = cosine * first + sine * second
            second_coeff[2] = -sine * first + cosine * second

            first = second_coeff[1].clone()
            second = second_coeff[3].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
            second_coeff[1] = cosine * first + sine * second
            second_coeff[3] = -sine * first + cosine * second

            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2_tail.copy_(second_coeff[3:4, 1:])
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                self.qk_second_center2,
=======
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                F.pad(self.qk_second_center2_tail, (1, 0)),
>>>>>>> REPLACE