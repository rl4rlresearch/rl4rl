MECHANISM: Coordinate-aligned second key channel

HYPOTHESIS: Fixing head 1’s second key projection to a residual coordinate will reduce the model from 996 to 991 parameters while retaining at least 99% accuracy, because it preserves two-dimensional content scoring and lets the learned embedding align the required source feature with that coordinate.

INTENDED_EDIT: Remove the five learned tail coefficients from head 1’s second key channel, reconstruct that channel with an identity prefix and zero tail, and project the virtual optimizer state back onto this constraint after every update.

EVIDENCE: Rank-two query/key routing achieved 99.93%, whereas rank one failed; unlike rank reduction, this preserves both score channels while challenging the assumption that every channel requires an independently learned full key mixture. Head 1’s strongly structured relative-bias pathway further makes a coordinate-aligned content channel plausible.

<<<<<<< SEARCH
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.score_dim,
                reduced_width - self.score_dim,
            )
        )
=======
        # The final flattened row is head 1's second key channel. Its
        # identity prefix remains active while its tail is fixed at zero.
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head * self.score_dim - 1,
                reduced_width - self.score_dim,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            key_basis = key_heads[..., : self.score_dim]
            self.query_weight.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            self.key_tail.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., self.score_dim :]
                )
            )
=======
            key_basis = key_heads[..., : self.score_dim]
            self.query_weight.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            canonical_key_tail = torch.linalg.solve(
                key_basis, key_heads[..., self.score_dim :]
            )
            flat_key_tail = canonical_key_tail.reshape(
                -1, self.d_model - 1 - self.score_dim
            )
            flat_key_tail[-1].zero_()
            self.key_tail.copy_(flat_key_tail[:-1])
            canonical_key_tail = flat_key_tail.view(
                self.n_head,
                self.score_dim,
                self.d_model - 1 - self.score_dim,
            )
            projected_key_heads = torch.cat(
                [
                    key_basis,
                    torch.matmul(key_basis, canonical_key_tail),
                ],
                dim=-1,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self._initial_key_weight = (
            key_heads.reshape(-1, self.d_model - 1).detach().clone()
        )
=======
        self._initial_key_weight = (
            projected_key_heads.reshape(
                -1, self.d_model - 1
            ).detach().clone()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_weight = torch.cat([key_prefix, self.key_tail], dim=-1)
=======
        key_tail = torch.cat(
            [
                self.key_tail,
                self.key_tail.new_zeros(
                    1, self.key_tail.shape[-1]
                ),
            ],
            dim=0,
        ).view(
            self.n_head,
            self.score_dim,
            reduced_x.shape[-1] - self.score_dim,
        )
        key_weight = torch.cat([key_prefix, key_tail], dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        canonical_key_tail_grad = (
            torch.zeros_like(key_tail_param)
            if key_tail_param.grad is None
            else key_tail_param.grad.detach()
        )
        canonical_bias_grad = (
            torch.zeros_like(bias_param)
            if bias_param.grad is None
            else bias_param.grad.detach()
        )

        n_head, head_dim, reduced_width = query_param.shape
        virtual_key_heads = state["key_weight"].view(
            n_head, head_dim, reduced_width
        )
        key_basis = virtual_key_heads[..., :head_dim]
=======
        stored_key_tail_grad = (
            torch.zeros_like(key_tail_param)
            if key_tail_param.grad is None
            else key_tail_param.grad.detach()
        )
        canonical_bias_grad = (
            torch.zeros_like(bias_param)
            if bias_param.grad is None
            else bias_param.grad.detach()
        )

        n_head, head_dim, reduced_width = query_param.shape
        tail_width = reduced_width - head_dim
        canonical_key_tail_grad = torch.cat(
            [
                stored_key_tail_grad,
                stored_key_tail_grad.new_zeros(1, tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        canonical_key_tail = torch.cat(
            [
                key_tail_param.detach(),
                key_tail_param.new_zeros(1, tail_width),
            ],
            dim=0,
        ).view(n_head, head_dim, tail_width)
        virtual_key_heads = state["key_weight"].view(
            n_head, head_dim, reduced_width
        )
        key_basis = virtual_key_heads[..., :head_dim]
>>>>>>> REPLACE

<<<<<<< SEARCH
                canonical_key_tail_grad,
                key_tail_param.transpose(-1, -2),
=======
                canonical_key_tail_grad,
                canonical_key_tail.transpose(-1, -2),
>>>>>>> REPLACE

<<<<<<< SEARCH
            key_basis = key_heads[..., :head_dim]
            query_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            key_tail_param.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., head_dim:]
                )
            )
            bias_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
=======
            key_basis = key_heads[..., :head_dim]
            query_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            canonical_key_tail = torch.linalg.solve(
                key_basis, key_heads[..., head_dim:]
            )
            flat_key_tail = canonical_key_tail.reshape(
                -1, reduced_width - head_dim
            )
            flat_key_tail[-1].zero_()
            key_tail_param.copy_(flat_key_tail[:-1])
            canonical_key_tail = flat_key_tail.view(
                n_head, head_dim, reduced_width - head_dim
            )
            projected_key_heads = torch.cat(
                [
                    key_basis,
                    torch.matmul(key_basis, canonical_key_tail),
                ],
                dim=-1,
            )
            state["key_weight"].copy_(
                projected_key_heads.reshape_as(state["key_weight"])
            )
            bias_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
>>>>>>> REPLACE