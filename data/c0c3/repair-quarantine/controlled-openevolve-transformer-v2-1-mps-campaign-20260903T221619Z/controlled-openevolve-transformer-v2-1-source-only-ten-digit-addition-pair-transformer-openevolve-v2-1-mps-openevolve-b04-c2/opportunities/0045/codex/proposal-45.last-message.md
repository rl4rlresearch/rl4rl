MECHANISM: Symmetric cross-pair intra-head value-basis gauge fixing

HYPOTHESIS: Reproducing the qualified 1,612-parameter design and applying its successful cross-pair rotation to the second attention head will yield 1,611 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Compact six value-projection coefficients by applying two disjoint rotations and one cross-pair rotation independently in each four-dimensional attention head, with exactly compensated output-projection rotations.

EVIDENCE: The first cross-pair extension reduced the fully disjoint 1,613-parameter design to 1,612 parameters with 99.98% accuracy; applying the identical remaining symmetry to the second head is the closest incremental test.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        if self.head_dim < 2:
            raise ValueError("value-basis compaction requires head dimension at least two")
        self.d_model = d_model

        # Fix two disjoint value pairs in each four-dimensional head, then
        # rotate the two remaining nonzero first-column coordinates together.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1))
            value_fixed_coordinates.append((head, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3))
                value_fixed_coordinates.append((head, 2))
                value_basis_rotations.append((head, 1, 3))
                value_fixed_coordinates.append((head, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                for head, local in value_fixed_coordinates
            )
        )

        self.qkv = nn.Linear(d_model, 3 * d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def compact_value_basis(self) -> None:
        # Attention applies the same mixing matrix to every value coordinate
        # within a head. Rotate two such coordinates, compensate with the
        # inverse rotation in the output projection, and gauge-fix one value
        # coefficient to zero without changing the initialized function.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()
            first_value = 2 * self.d_model
            second_value = first_value + 1

            a = qkv_weight[first_value, 0]
            b = qkv_weight[second_value, 0]
            norm = torch.hypot(a, b)
            cosine = b / norm
            sine = a / norm

            row0 = qkv_weight[first_value].clone()
            row1 = qkv_weight[second_value].clone()
            qkv_weight[first_value] = cosine * row0 - sine * row1
            qkv_weight[second_value] = sine * row0 + cosine * row1

            col0 = proj_weight[:, 0].clone()
            col1 = proj_weight[:, 1].clone()
            proj_weight[:, 0] = cosine * col0 - sine * col1
            proj_weight[:, 1] = sine * col0 + cosine * col1
            self.proj.weight.copy_(proj_weight)

            flat_weight = qkv_weight.reshape(-1)
            fixed_index = first_value * self.d_model
            compact_weight = torch.cat(
                (flat_weight[:fixed_index], flat_weight[fixed_index + 1 :])
            )
        self.qkv.weight = nn.Parameter(compact_weight)
=======
    def compact_value_basis(self) -> None:
        # Attention mixes value coordinates identically within each head.
        # Rotate each selected pair, counter-rotate the corresponding output
        # columns, and omit the coefficient made exactly zero.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for head, first_local, second_local in self.value_basis_rotations:
                first_value = (
                    2 * self.d_model + head * self.head_dim + first_local
                )
                second_value = (
                    2 * self.d_model + head * self.head_dim + second_local
                )
                first_column = head * self.head_dim + first_local
                second_column = head * self.head_dim + second_local

                a = qkv_weight[first_value, 0]
                b = qkv_weight[second_value, 0]
                norm = torch.hypot(a, b)
                cosine = b / norm
                sine = a / norm

                row0 = qkv_weight[first_value].clone()
                row1 = qkv_weight[second_value].clone()
                qkv_weight[first_value] = cosine * row0 - sine * row1
                qkv_weight[second_value] = sine * row0 + cosine * row1

                col0 = proj_weight[:, first_column].clone()
                col1 = proj_weight[:, second_column].clone()
                proj_weight[:, first_column] = cosine * col0 - sine * col1
                proj_weight[:, second_column] = sine * col0 + cosine * col1

            self.proj.weight.copy_(proj_weight)
            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
                pieces.append(flat_weight[start:fixed_index])
                start = fixed_index + 1
            pieces.append(flat_weight[start:])
            compact_weight = torch.cat(pieces)
        self.qkv.weight = nn.Parameter(compact_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        fixed_index = 2 * self.d_model * self.d_model
        qkv_weight = torch.cat(
            (
                self.qkv.weight[:fixed_index],
                self.qkv.weight.new_zeros(1),
                self.qkv.weight[fixed_index:],
            )
        ).view(3 * self.d_model, self.d_model)
=======
        weight_pieces = []
        compact_start = 0
        for removed, fixed_index in enumerate(self.value_fixed_indices):
            compact_index = fixed_index - removed
            weight_pieces.append(self.qkv.weight[compact_start:compact_index])
            weight_pieces.append(self.qkv.weight.new_zeros(1))
            compact_start = compact_index
        weight_pieces.append(self.qkv.weight[compact_start:])
        qkv_weight = torch.cat(weight_pieces).view(
            3 * self.d_model, self.d_model
        )
>>>>>>> REPLACE