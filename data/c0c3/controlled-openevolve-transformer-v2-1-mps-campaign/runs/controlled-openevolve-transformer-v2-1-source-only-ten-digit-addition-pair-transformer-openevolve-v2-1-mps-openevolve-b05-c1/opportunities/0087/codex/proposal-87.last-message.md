MECHANISM: Third function-preserving query/key rotation in the first attention head

HYPOTHESIS: Fixing the remaining centered rotational degree of freedom in the first head will reduce the model from 1,269 to 1,268 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate the first head’s center-1/center-2 plane during initialization, omit center-2’s second input coefficient, and reconstruct both fixed coefficients as zeros during forward passes.

EVIDENCE: The first head retained 100% accuracy after its second sequential rotation gauge at 1,269 parameters, whereas applying that gauge to the second head fell to 73.74%; therefore the smallest supported next test is the remaining rotation in the head that has tolerated both prior constraints.

<<<<<<< SEARCH
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
=======
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
            coeff[1] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2_tail.copy_(coeff[3:4, 1:])
=======
            coeff[1] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

            first = coeff[2].clone()
            second = coeff[3].clone()
            radius = torch.sqrt(first[1].square() + second[1].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[1] / radius
            sine = second[1] / radius
            coeff[2] = cosine * first + sine * second
            coeff[3] = -sine * first + cosine * second

            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2_tail.copy_(coeff[3:4, 2:])
>>>>>>> REPLACE

<<<<<<< SEARCH
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (1, 0)),
=======
                F.pad(self.qk_center1_tail, (1, 0)),
                F.pad(self.qk_center2_tail, (2, 0)),
>>>>>>> REPLACE