MECHANISM: Complete centered orthogonal gauge fixing in the first attention head

HYPOTHESIS: Removing the first free coefficient of the first head’s final centered direction will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy, because an additional centered-basis rotation preserves all query-key dot products and initialization-time model outputs.

INTENDED_EDIT: Apply the remaining centered rotation in the first head, constrain `qk_center2_tail` to begin with two zeros, and store only its final five coefficients.

EVIDENCE: The 1,268-parameter model achieved 100% accuracy after two centered rotations in the first head. This extends the same exact query-key symmetry without the functionally distinct parameter coupling that caused the 1,267-parameter LayerNorm-sharing attempts to fail.

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