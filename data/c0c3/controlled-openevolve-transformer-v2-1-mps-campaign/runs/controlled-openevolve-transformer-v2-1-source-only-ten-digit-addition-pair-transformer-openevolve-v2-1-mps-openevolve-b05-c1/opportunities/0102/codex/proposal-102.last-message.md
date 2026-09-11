MECHANISM: Complete first-head centered rotation gauge

HYPOTHESIS: Fixing the remaining rotationally redundant coefficient in the first attention head will reduce the model from 1,266 to 1,265 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Rotate the first head’s center-1/center-2 plane at input coordinate 1, omit center-2’s coordinate-1 coefficient, and reconstruct it as zero during forward passes.

EVIDENCE: Two rotation constraints in the first head retained 100% accuracy, whereas further constraints in the second head failed; completing the exact centered-subspace gauge in the already-tolerant first head tests a distinct 1,265-parameter reduction without the runtime overhead that caused repeated third null-reuse timeouts.

<<<<<<< SEARCH
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
=======
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2_tail = nn.Parameter(torch.empty(1, in_features - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            first = coeff[1].clone()
            second = coeff[3].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
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