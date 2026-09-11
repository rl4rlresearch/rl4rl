MECHANISM: Learned relative-kernel causal attention

HYPOTHESIS: Replacing content-dependent query–key scores with two learned causal relative-distance kernels will reduce the model from 1,041 to 925 parameters while retaining at least 99% accuracy, because fixed-format addition requires distinct positional routes but not necessarily token-dependent routing.

INTENDED_EDIT: Remove all learned query and key maps and their bias after initialization; retain learned per-head relative biases, the shared learned value map, and distinct independently scaled residual slots.

EVIDENCE: Disjoint head-slot routing achieved 99.96%, while merging routes collapsed accuracy to 0.01%, identifying route identity as load-bearing. The shared value map succeeded, and extensive cross-head relative-bias sharing also retained accuracy, making learned relative-distance kernels a plausible compact source of the two required routes.

<<<<<<< SEARCH
    def gauge_fix_qkv(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            weight = self.qkv.weight
            q_weight = weight[:d_model]
            k_weight = weight[d_model : 2 * d_model]
            v_weight = weight[2 * d_model :]
            q_weight = q_weight - q_weight.mean(dim=1, keepdim=True)
            q_coeff = q_weight @ self.proj_basis
            k_weight = k_weight - k_weight.mean(dim=1, keepdim=True)
            k_coeff = k_weight @ self.proj_basis

            # A joint orthogonal rotation of a head's query and key features
            # leaves every dot-product attention score unchanged. Rotate the
            # first two features of the first head so one query coefficient is
            # exactly zero, then omit that fixed gauge coordinate below.
            rotation_column = torch.linalg.vector_norm(
                q_coeff[:2], dim=0
            ).argmax()
            a = q_coeff[0, rotation_column]
            b = q_coeff[1, rotation_column]
            radius = torch.sqrt(a.square() + b.square())
            rotation = torch.stack(
                (
                    torch.stack((b / radius, -a / radius)),
                    torch.stack((a / radius, b / radius)),
                )
            )
            q_coeff[:2] = rotation @ q_coeff[:2]
            k_coeff[:2] = rotation @ k_coeff[:2]
            self.qkv.bias[:2].copy_(
                rotation @ self.qkv.bias[:2]
            )
            q_coeff[0, rotation_column] = 0.0

            # Apply the same independent orthogonal gauge fix to the second
            # head, preserving all of its attention scores exactly.
            second_rotation_column = torch.linalg.vector_norm(
                q_coeff[self.head_dim : self.head_dim + 2], dim=0
            ).argmax()
            a = q_coeff[self.head_dim, second_rotation_column]
            b = q_coeff[self.head_dim + 1, second_rotation_column]
            radius = torch.sqrt(a.square() + b.square())
            second_rotation = torch.stack(
                (
                    torch.stack((b / radius, -a / radius)),
                    torch.stack((a / radius, b / radius)),
                )
            )
            q_coeff[self.head_dim : self.head_dim + 2] = (
                second_rotation
                @ q_coeff[self.head_dim : self.head_dim + 2]
            )
            k_coeff[self.head_dim : self.head_dim + 2] = (
                second_rotation
                @ k_coeff[self.head_dim : self.head_dim + 2]
            )
            self.qkv.bias[
                self.head_dim : self.head_dim + 2
            ].copy_(
                second_rotation
                @ self.qkv.bias[self.head_dim : self.head_dim + 2]
            )
            q_coeff[
                self.head_dim, second_rotation_column
            ] = 0.0
=======
    def compress_to_static_attention_value(self) -> None:
        with torch.no_grad():
            d_model = self.n_head * self.head_dim
            # Fixed-format addition can learn its two causal routing patterns
            # directly as relative-distance softmax kernels. Discard the
            # content-dependent query and key maps while retaining a learned
            # semantic value map shared by the distinct routes.
            v_weight = self.qkv.weight[2 * d_model :]
>>>>>>> REPLACE

<<<<<<< SEARCH
            # Each head has an independent reciprocal query-key scaling
            # symmetry. Retain the first head's fixed key coordinate and the
            # complementary second-head query pivot, while excluding both
            # orthogonally fixed zero query coordinates.
            q_flat = q_coeff.reshape(-1)
            second_head_start = self.head_dim * (d_model - 1)
            q_pivot_index = second_head_start + q_flat[
                second_head_start:
            ].abs().argmax()
            first_q_zero_index = rotation_column
            second_q_zero_index = (
                second_head_start + second_rotation_column
            )
            q_all_coordinates = torch.arange(
                q_flat.numel(), device=q_flat.device
            )
            q_tail_coordinates = q_all_coordinates[
                (q_all_coordinates != q_pivot_index)
                & (q_all_coordinates != first_q_zero_index)
                & (q_all_coordinates != second_q_zero_index)
            ]
            q_coordinate_order = torch.cat(
                (
                    q_pivot_index.unsqueeze(0),
                    first_q_zero_index.unsqueeze(0),
                    second_q_zero_index.unsqueeze(0),
                    q_tail_coordinates,
                )
            )
            self.register_buffer(
                "q_pivot", q_flat[q_pivot_index].detach().clone()
            )
            self.register_buffer(
                "q_inverse_order", torch.argsort(q_coordinate_order)
            )

            k_flat = k_coeff.reshape(-1)
            first_head_size = self.head_dim * (d_model - 1)
            k_pivot_index = k_flat[:first_head_size].abs().argmax()
            k_all_coordinates = torch.arange(
                k_flat.numel(), device=k_flat.device
            )
            k_tail_coordinates = k_all_coordinates[
                k_all_coordinates != k_pivot_index
            ]
            k_coordinate_order = torch.cat(
                (k_pivot_index.unsqueeze(0), k_tail_coordinates)
            )
            self.register_buffer(
                "k_pivot", k_flat[k_pivot_index].detach().clone()
            )
            self.register_buffer(
                "k_inverse_order", torch.argsort(k_coordinate_order)
            )
            self.qkv.weight = nn.Parameter(
                torch.cat(
                    (
                        q_flat[q_tail_coordinates],
                        k_flat[k_tail_coordinates],
                        v_flat[tail_coordinates],
                    )
                ).clone()
            )
=======
            self.qkv.weight = nn.Parameter(
                v_flat[tail_coordinates].clone()
            )
            self.qkv.bias = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        matrix_size = d_model * (d_model - 1)
        q_flat = torch.cat(
            (
                self.q_pivot.unsqueeze(0),
                self.qkv.weight.new_zeros(2),
                self.qkv.weight[: matrix_size - 3],
            )
        )
        q_coeff = q_flat[self.q_inverse_order].view(
            d_model, d_model - 1
        )
        q_weight = q_coeff @ self.proj_basis.T
        k_flat = torch.cat(
            (
                self.k_pivot.unsqueeze(0),
                self.qkv.weight[
                    matrix_size - 3 : 2 * matrix_size - 4
                ],
            )
        )
        k_coeff = k_flat[self.k_inverse_order].view(
            d_model, d_model - 1
        )
        k_weight = k_coeff @ self.proj_basis.T
        v_flat = torch.cat(
            (
                self.v_pivot.unsqueeze(0),
                self.qkv.weight[2 * matrix_size - 4 :],
            )
        )
        v_coeff = v_flat[self.v_inverse_order].view(
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        q = F.linear(x, q_weight, self.qkv.bias)
        k = F.linear(x, k_weight)
        v = F.linear(x, v_weight)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        bsz, seqlen, d_model = x.shape
        v_flat = torch.cat(
            (
                self.v_pivot.unsqueeze(0),
                self.qkv.weight,
            )
        )
        v_coeff = v_flat[self.v_inverse_order].view(
            self.head_dim, d_model - 1
        )
        v_weight = v_coeff @ self.proj_basis.T
        v = F.linear(x, v_weight)
        v = v.unsqueeze(1).expand(-1, self.n_head, -1, -1)

        att = x.new_zeros(
            bsz, self.n_head, seqlen, seqlen
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            block.attn.gauge_fix_qkv()
=======
            block.attn.compress_to_static_attention_value()
>>>>>>> REPLACE