MECHANISM: Symmetric per-head orthogonal attention gauge fixing

HYPOTHESIS: Fixing a second centered rotation in the second attention head will reduce the model from 1,268 to 1,267 parameters while retaining at least 99% accuracy, because the rotation preserves all query-key dot products and the fresh-initialization function.

INTENDED_EDIT: Rotate the second head’s center-0 and center-2 coefficients to zero the first center-2 coefficient, then store only its remaining six coefficients.

EVIDENCE: The 1,268-parameter design reached 100% accuracy while already applying this two-rotation chart to the first head. Unlike the failed dual LayerNorm-scale reuse at 15.95% and 54.41%, this removes an exact query-key orthogonal gauge without coupling functionally distinct optimizer coordinates.

<<<<<<< SEARCH
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
=======
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
            second_coeff[1] = cosine * first + sine * second
            second_coeff[2] = -sine * first + cosine * second

            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
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
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.qk_second_common,
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                self.qk_second_center2,
=======
                self.qk_second_common,
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                F.pad(self.qk_second_center2_tail, (1, 0)),
>>>>>>> REPLACE