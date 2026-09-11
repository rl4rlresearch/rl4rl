MECHANISM: Residual-output common-mode quotient

HYPOTHESIS: Extending the qualified 1,609-parameter value triangularization by removing the function-null uniform output component of one attention projection column will yield 1,608 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified three-first-column/two-second-column value alignment, then parameterize the first attention output-projection column in the same seven-dimensional zero-sum basis already used for its bias and train it with full-coordinate quotient moments.

EVIDENCE: Reference Design 3 achieved 99.87% at 1,609 parameters, whereas further value triangularization collapsed to 74.15% and query–key co-rotation to 14.2%; the already-qualified projection-bias common-mode quotient shows that removing an attention-output uniform direction is a distinct, stable mechanism.

<<<<<<< SEARCH
        # First fix two disjoint pairs in every four-dimensional head. In the
        # first head, the remaining two nonzero first-column coordinates can
        # then be rotated together without disturbing either existing zero.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1))
            value_fixed_coordinates.append((head, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3))
                value_fixed_coordinates.append((head, 2))
                if head == 0:
                    value_basis_rotations.append((head, 1, 3))
                    value_fixed_coordinates.append((head, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                for head, local in value_fixed_coordinates
            )
        )
=======
        # Align each head's first value-weight column with one basis vector.
        # The resulting three-dimensional zero subspace in the first head can
        # then be triangularized through two second-column rotations.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1, 0))
            value_fixed_coordinates.append((head, 0, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3, 0))
                value_fixed_coordinates.append((head, 2, 0))
                value_basis_rotations.append((head, 1, 3, 0))
                value_fixed_coordinates.append((head, 1, 0))
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                + input_column
                for head, local, input_column in value_fixed_coordinates
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            for (
                head,
                first_local,
                second_local,
                input_column,
            ) in self.value_basis_rotations:
                first_value = (
                    2 * self.d_model + head * self.head_dim + first_local
                )
                second_value = (
                    2 * self.d_model + head * self.head_dim + second_local
                )
                first_column = head * self.head_dim + first_local
                second_column = head * self.head_dim + second_local

                a = qkv_weight[first_value, input_column]
                b = qkv_weight[second_value, input_column]
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
                pieces.append(flat_weight[start:fixed_index])
                start = fixed_index + 1
            pieces.append(flat_weight[start:])
            compact_weight = torch.cat(pieces)

            # Any uniform output-row component of a projection column only
            # adds a per-token common-mode residual shift. Subsequent
            # LayerNorms erase it, so quotient one column by that direction.
            compact_proj_column = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, 0]
            )
            remaining_proj_weight = proj_weight[:, 1:].clone()
        self.qkv.weight = nn.Parameter(compact_weight)
        self.proj_first_column = nn.Parameter(compact_proj_column)
        self.proj.weight = nn.Parameter(remaining_proj_weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        proj_bias = self.proj_bias_basis @ self.proj.bias
        y = F.linear(y, self.proj.weight, proj_bias)
=======
        proj_bias = self.proj_bias_basis @ self.proj.bias
        proj_first_column = (
            self.proj_bias_basis @ self.proj_first_column
        ).unsqueeze(1)
        proj_weight = torch.cat(
            (proj_first_column, self.proj.weight), dim=1
        )
        y = F.linear(y, proj_weight, proj_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
=======
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_first_column, blk.attn.proj_bias_basis),
>>>>>>> REPLACE