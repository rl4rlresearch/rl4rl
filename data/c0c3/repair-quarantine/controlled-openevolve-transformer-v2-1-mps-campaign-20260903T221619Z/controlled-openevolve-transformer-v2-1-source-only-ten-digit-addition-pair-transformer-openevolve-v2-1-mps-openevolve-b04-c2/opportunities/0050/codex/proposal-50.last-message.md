MECHANISM: Compensated query–key basis gauge fixing

HYPOTHESIS: Reproducing the qualified 1,609-parameter value triangularization and replacing the failed third-column value constraint with one exact query–key co-rotation will yield 1,608 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Add the qualified second residual value rotation in head 0, then co-rotate one query/key channel pair and omit the resulting zero key coefficient.

EVIDENCE: Reference Design 3 achieved 99.87% at 1,609 parameters, while extending its increasingly constrained value subspace to 1,608 collapsed to 74.15%; an independent query–key orthogonal gauge avoids that failed value-path allocation.

<<<<<<< SEARCH
        # Align each head's first value-weight column with one basis vector.
        # In the first head, two of the resulting zero coordinates retain an
        # orthogonal stabilizer that can also eliminate one second-column
        # coefficient without disturbing any first-column zero.
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
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                + input_column
                for head, local, input_column in value_fixed_coordinates
            )
        )
=======
        # Align each head's first value-weight column with one basis vector.
        # Fully triangularize the surviving second-column value subspace in
        # head zero, as in the qualified 1,609-parameter design.
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

        # Query and key coordinates may be co-rotated within a head without
        # changing their dot products. Use one such rotation to eliminate a
        # key-weight coefficient independently of the value constraints.
        self.query_key_basis_rotations = ((0, 0, 1, 0),)
        qkv_fixed_indices = [
            (2 * d_model + head * self.head_dim + local) * d_model
            + input_column
            for head, local, input_column in value_fixed_coordinates
        ]
        qkv_fixed_indices.extend(
            (d_model + head * self.head_dim + local) * d_model
            + input_column
            for head, local, _, input_column in self.query_key_basis_rotations
        )
        self.qkv_fixed_indices = tuple(sorted(qkv_fixed_indices))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def compact_value_basis(self) -> None:
        # Attention mixes value coordinates identically within each head.
        # Rotate each selected pair, counter-rotate the corresponding output
        # columns, and omit the coefficient made exactly zero.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for (
=======
    def compact_attention_bases(self) -> None:
        # Apply exact within-head basis changes, then omit the coefficients
        # selected as gauge coordinates.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            q_bias = self.qkv.bias.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for (
                head,
                first_local,
                second_local,
                input_column,
            ) in self.query_key_basis_rotations:
                first_query = head * self.head_dim + first_local
                second_query = head * self.head_dim + second_local
                first_key = self.d_model + first_query
                second_key = self.d_model + second_query

                a = qkv_weight[first_key, input_column]
                b = qkv_weight[second_key, input_column]
                norm = torch.hypot(a, b)
                cosine = b / norm
                sine = a / norm

                query_row0 = qkv_weight[first_query].clone()
                query_row1 = qkv_weight[second_query].clone()
                qkv_weight[first_query] = cosine * query_row0 - sine * query_row1
                qkv_weight[second_query] = sine * query_row0 + cosine * query_row1

                key_row0 = qkv_weight[first_key].clone()
                key_row1 = qkv_weight[second_key].clone()
                qkv_weight[first_key] = cosine * key_row0 - sine * key_row1
                qkv_weight[second_key] = sine * key_row0 + cosine * key_row1

                bias0 = q_bias[first_query].clone()
                bias1 = q_bias[second_query].clone()
                q_bias[first_query] = cosine * bias0 - sine * bias1
                q_bias[second_query] = sine * bias0 + cosine * bias1

            for (
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.proj.weight.copy_(proj_weight)
            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
=======
            self.qkv.bias.copy_(q_bias)
            self.proj.weight.copy_(proj_weight)
            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.qkv_fixed_indices:
>>>>>>> REPLACE

<<<<<<< SEARCH
        for removed, fixed_index in enumerate(self.value_fixed_indices):
=======
        for removed, fixed_index in enumerate(self.qkv_fixed_indices):
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.blocks:
            block.attn.compact_value_basis()
=======
        for block in self.blocks:
            block.attn.compact_attention_bases()
>>>>>>> REPLACE