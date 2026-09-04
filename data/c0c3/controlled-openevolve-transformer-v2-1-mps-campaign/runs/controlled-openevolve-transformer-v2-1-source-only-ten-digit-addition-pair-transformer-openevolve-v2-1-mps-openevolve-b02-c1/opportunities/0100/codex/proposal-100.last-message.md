MECHANISM: Head-shared canonical value subspace

HYPOTHESIS: Sharing the learned canonical value map across both attention heads will reduce the model from 985 to 973 parameters while retaining at least 99% accuracy, because addition needs distinct attention routes but can encode the digits retrieved by those routes in one common learned feature space.

INTENDED_EDIT: Store one learned 4-by-3 value tail, use it for both heads, initialize it by projecting the original head-specific maps onto their shared mean, and project the virtual optimizer state back onto that shared value subspace after every update.

EVIDENCE: Fixing five coefficients of one key channel preserved 99.94% accuracy while retaining both score dimensions, showing that head-local content projections contain substantial redundancy; unlike the failed rank-one change, this patch preserves both rank-two routing mechanisms, their relative biases, and their independent output readouts.

<<<<<<< SEARCH
        self.value_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.head_dim,
                reduced_width - self.head_dim,
            )
        )
=======
        # Both position-specialized heads read tokens through one learned
        # canonical value subspace; their routing and output readouts remain
        # independent.
        self.value_tail = nn.Parameter(
            torch.empty(
                1,
                self.head_dim,
                reduced_width - self.head_dim,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            value_basis = value_heads[..., : self.head_dim]
            self.value_tail.copy_(
                torch.linalg.solve(
                    value_basis, value_heads[..., self.head_dim :]
                )
            )

        # Consumed by QuotientAdamW to preserve virtual rank-three updates.
        self._initial_query_weight = (
            query_heads.reshape(-1, self.d_model - 1).detach().clone()
        )
        self._initial_key_weight = (
            projected_key_heads.reshape(
                -1, self.d_model - 1
            ).detach().clone()
        )
        self._initial_query_bias = (
            bias_heads.reshape(-1).detach().clone()
        )
        self._initial_value_weight = value_weight.detach().clone()
=======
            value_basis = value_heads[..., : self.head_dim]
            canonical_value_tail = torch.linalg.solve(
                value_basis, value_heads[..., self.head_dim :]
            )
            shared_value_tail = canonical_value_tail.mean(
                dim=0, keepdim=True
            )
            self.value_tail.copy_(shared_value_tail)
            projected_value_heads = torch.cat(
                [
                    value_basis,
                    torch.matmul(
                        value_basis,
                        shared_value_tail.expand(
                            self.n_head, -1, -1
                        ),
                    ),
                ],
                dim=-1,
            )
            projected_effective_value = projected_value_heads.reshape(
                self.d_model, self.d_model - 1
            )
            value_reference = value_weight[:, -1:]
            projected_value_weight = torch.cat(
                [
                    projected_effective_value + value_reference,
                    value_reference,
                ],
                dim=-1,
            )

        # Consumed by QuotientAdamW to preserve virtual rank-three updates.
        self._initial_query_weight = (
            query_heads.reshape(-1, self.d_model - 1).detach().clone()
        )
        self._initial_key_weight = (
            projected_key_heads.reshape(
                -1, self.d_model - 1
            ).detach().clone()
        )
        self._initial_query_bias = (
            bias_heads.reshape(-1).detach().clone()
        )
        self._initial_value_weight = (
            projected_value_weight.detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv.value_tail.copy_(
            torch.linalg.solve(
                value_basis, value_heads[..., self.head_dim :]
            )
        )
=======
        self.qkv.value_tail.copy_(
            torch.linalg.solve(
                value_basis, value_heads[..., self.head_dim :]
            ).mean(dim=0, keepdim=True)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_weight = torch.cat(
            [value_prefix, self.value_tail], dim=-1
        ).reshape(self.d_model, self.d_model - 1)
=======
        value_weight = torch.cat(
            [
                value_prefix,
                self.value_tail.expand(self.n_head, -1, -1),
            ],
            dim=-1,
        ).reshape(self.d_model, self.d_model - 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
                {
                    "step": 0,
                    "full_weight": full_weight,
                    "proj_weight": virtual_proj_weight,
=======
                {
                    "step": 0,
                    "n_head": qkv.n_head,
                    "full_weight": full_weight,
                    "proj_weight": virtual_proj_weight,
>>>>>>> REPLACE

<<<<<<< SEARCH
        n_head, head_dim, tail_width = weight_param.shape
        reduced_width = head_dim + tail_width
=======
        _, head_dim, tail_width = weight_param.shape
        n_head = state["n_head"]
        reduced_width = head_dim + tail_width
>>>>>>> REPLACE

<<<<<<< SEARCH
        solved_tail_grad = torch.linalg.solve(
            value_basis.transpose(-1, -2),
            canonical_tail_grad,
        )
=======
        per_head_tail_grad = canonical_tail_grad.expand(
            n_head, -1, -1
        ) / n_head
        solved_tail_grad = torch.linalg.solve(
            value_basis.transpose(-1, -2),
            per_head_tail_grad,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_prefix_grad = value_prefix_grad - torch.matmul(
            solved_tail_grad,
            weight_param.transpose(-1, -2),
        )
=======
        value_prefix_grad = value_prefix_grad - torch.matmul(
            solved_tail_grad,
            weight_param.expand(
                n_head, -1, -1
            ).transpose(-1, -2),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            n_head, head_dim, _ = weight.shape
            value_heads = effective_weight.view(
                n_head, head_dim, -1
            )
            value_basis = value_heads[..., :head_dim]
            weight.copy_(
                torch.linalg.solve(
                    value_basis, value_heads[..., head_dim:]
                )
            )
=======
            n_head = state["n_head"]
            head_dim = weight.shape[1]
            value_heads = effective_weight.view(
                n_head, head_dim, -1
            )
            value_basis = value_heads[..., :head_dim]
            canonical_value_tail = torch.linalg.solve(
                value_basis, value_heads[..., head_dim:]
            )
            shared_value_tail = canonical_value_tail.mean(
                dim=0, keepdim=True
            )
            weight.copy_(shared_value_tail)

            projected_value_heads = torch.cat(
                [
                    value_basis,
                    torch.matmul(
                        value_basis,
                        shared_value_tail.expand(n_head, -1, -1),
                    ),
                ],
                dim=-1,
            )
            projected_effective_weight = projected_value_heads.reshape_as(
                effective_weight
            )
            scaled_reference = scaled_weight[:, -1:]
            projected_scaled_weight = torch.cat(
                [
                    projected_effective_weight + scaled_reference,
                    scaled_reference,
                ],
                dim=1,
            )
            state["full_weight"].copy_(
                projected_scaled_weight
                / state["scale"].unsqueeze(0)
            )
>>>>>>> REPLACE