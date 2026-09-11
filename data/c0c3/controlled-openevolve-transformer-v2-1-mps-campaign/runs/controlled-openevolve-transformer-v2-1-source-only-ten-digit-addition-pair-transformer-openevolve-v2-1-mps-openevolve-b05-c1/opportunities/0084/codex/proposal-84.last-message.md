MECHANISM: One rotational gauge per attention head

HYPOTHESIS: Applying the proven single-rotation query/key gauge independently to the second head will reduce the verified model from 1,271 to 1,270 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the second head’s dense tied query/key weights with the same one-parameter-smaller, function-preserving centered-rotation chart already used by the first head.

EVIDENCE: One rotation in the first head retained 100% accuracy at 1,271 parameters, whereas imposing all six rotations simultaneously reached only 55.34%; adding one analogous constraint to the untouched head is the smallest supported continuation.

<<<<<<< SEARCH
class SingleRotationGaugeQKV(nn.Module):
    """Tied query/key and value map with one centered rotation fixed."""
=======
class SingleRotationGaugeQKV(nn.Module):
    """Tied query/key and value map with one centered rotation fixed per head."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qk_center2 = nn.Parameter(torch.empty(1, in_features))
        self.qk_rest = nn.Parameter(torch.empty(d_model - head_dim, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
=======
        self.qk_center2 = nn.Parameter(torch.empty(1, in_features))
        self.qk_second_common = nn.Parameter(torch.empty(1, in_features))
        self.qk_second_center0 = nn.Parameter(torch.empty(1, in_features))
        self.qk_second_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_second_center2 = nn.Parameter(torch.empty(1, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2.copy_(coeff[3:4])
            nn.init.normal_(self.qk_rest, mean=0.0, std=0.02)
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
=======
            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2.copy_(coeff[3:4])

            dense_second = torch.empty_like(
                self.qk_second_common.expand(self.head_basis.size(0), -1)
            )
            nn.init.normal_(dense_second, mean=0.0, std=0.02)
            second_coeff = self.head_basis.transpose(0, 1) @ dense_second

            first = second_coeff[1].clone()
            second = second_coeff[2].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
            second_coeff[1] = cosine * first + sine * second
            second_coeff[2] = -sine * first + cosine * second

            self.qk_second_common.copy_(second_coeff[:1])
            self.qk_second_center0.copy_(second_coeff[1:2])
            self.qk_second_center1_tail.copy_(second_coeff[2:3, 1:])
            self.qk_second_center2.copy_(second_coeff[3:4])
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_head = self.head_basis @ head_coeff
        qk_weight = torch.cat((first_head, self.qk_rest), dim=0)
=======
        first_head = self.head_basis @ head_coeff
        second_head_coeff = torch.cat(
            (
                self.qk_second_common,
                self.qk_second_center0,
                F.pad(self.qk_second_center1_tail, (1, 0)),
                self.qk_second_center2,
            ),
            dim=0,
        )
        second_head = self.head_basis @ second_head_coeff
        qk_weight = torch.cat((first_head, second_head), dim=0)
>>>>>>> REPLACE