MECHANISM: Optimizer-preserving per-head value/output basis gauge quotient

HYPOTHESIS: Canonicalizing each head’s rank-four value projection to an identity basis will reduce the model from 1,251 to 1,219 parameters while retaining at least 99% accuracy, because it preserves the exact attention output and reconstructs the successful virtual value/output AdamW gradients through the gauge transformation.

INTENDED_EDIT: Replace the 56 stored value-projection parameters with 24 canonical tail parameters, transform each head’s output-projection columns into the matching basis, and retain/update the original virtual value and output factors inside the quotient-aware optimizer.

EVIDENCE: The analogous optimizer-preserving Q/K basis canonicalization removed 24 parameters and achieved 99.98% accuracy; the value/output factors have the same exact per-head four-dimensional change-of-basis invariance, while the 1,397-parameter joint attention folding result shows that reconstructing coupled value, projection, and bias gradients can retain 99.93% accuracy.

<<<<<<< SEARCH
        self.query_bias = nn.Parameter(
            torch.empty(n_head, self.head_dim)
        )
        self.value_weight = nn.Parameter(torch.empty(d_model, reduced_width))
=======
        self.query_bias = nn.Parameter(
            torch.empty(n_head, self.head_dim)
        )
        self.value_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.head_dim,
                reduced_width - self.head_dim,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_weight = full_weight[2 * self.d_model :]
        query_weight = (
            full_query_weight[:, :-1] - full_query_weight[:, -1:]
        )
        key_weight = (
            full_key_weight[:, :-1] - full_key_weight[:, -1:]
        )
        query_bias = full_weight.new_zeros(self.d_model)
        query_heads = query_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        key_heads = key_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        bias_heads = query_bias.view(self.n_head, self.head_dim)

        with torch.no_grad():
            key_basis = key_heads[..., : self.head_dim]
            self.query_weight.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            self.key_tail.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., self.head_dim :]
                )
            )
            self.query_bias.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
            self.value_weight.copy_(
                value_weight[:, :-1] - value_weight[:, -1:]
            )

        # Consumed by QuotientAdamW to preserve the virtual factor updates.
        self._initial_query_weight = query_weight.detach().clone()
        self._initial_key_weight = key_weight.detach().clone()
        self._initial_query_bias = query_bias.detach().clone()
        self._initial_value_weight = value_weight.detach().clone()
=======
        value_weight = full_weight[2 * self.d_model :]
        query_weight = (
            full_query_weight[:, :-1] - full_query_weight[:, -1:]
        )
        key_weight = (
            full_key_weight[:, :-1] - full_key_weight[:, -1:]
        )
        reduced_value_weight = (
            value_weight[:, :-1] - value_weight[:, -1:]
        )
        query_bias = full_weight.new_zeros(self.d_model)
        query_heads = query_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        key_heads = key_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        value_heads = reduced_value_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        bias_heads = query_bias.view(self.n_head, self.head_dim)

        with torch.no_grad():
            key_basis = key_heads[..., : self.head_dim]
            self.query_weight.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            self.key_tail.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., self.head_dim :]
                )
            )
            self.query_bias.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
            value_basis = value_heads[..., : self.head_dim]
            self.value_tail.copy_(
                torch.linalg.solve(
                    value_basis,
                    value_heads[..., self.head_dim :],
                )
            )

        # Consumed by QuotientAdamW to preserve the virtual factor updates.
        self._initial_query_weight = query_weight.detach().clone()
        self._initial_key_weight = key_weight.detach().clone()
        self._initial_query_bias = query_bias.detach().clone()
        self._initial_value_weight = value_weight.detach().clone()
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = torch.einsum("bhtd,bhsd->bhts", q, k)
        v = F.linear(reduced_x, self.value_weight)
        return att, v
=======
        att = torch.einsum("bhtd,bhsd->bhts", q, k)
        value_prefix = torch.eye(
            self.head_dim,
            device=reduced_x.device,
            dtype=reduced_x.dtype,
        ).expand(self.n_head, -1, -1)
        value_weight = torch.cat(
            [value_prefix, self.value_tail], dim=-1
        ).reshape(self.d_model, self.d_model - 1)
        v = F.linear(reduced_x, value_weight)
        return att, v
>>>>>>> REPLACE

<<<<<<< SEARCH
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self, x: torch.Tensor, normalized_x: torch.Tensor
    ) -> torch.Tensor:
=======
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    @torch.no_grad()
    def canonicalize_value_basis(self) -> None:
        full_value_weight = self.qkv._initial_value_weight
        reduced_value_weight = (
            full_value_weight[:, :-1] - full_value_weight[:, -1:]
        )
        value_heads = reduced_value_weight.view(
            self.n_head, self.head_dim, reduced_value_weight.shape[1]
        )
        value_basis = value_heads[..., : self.head_dim]

        original_proj_weight = self.proj.weight.detach().clone()
        proj_heads = original_proj_weight.view(
            original_proj_weight.shape[0],
            self.n_head,
            self.head_dim,
        )
        canonical_proj = torch.einsum(
            "rhd,hde->rhe", proj_heads, value_basis
        )
        self.proj.weight.copy_(
            canonical_proj.reshape_as(self.proj.weight)
        )
        self.qkv._initial_proj_weight = original_proj_weight

    def forward(
        self, x: torch.Tensor, normalized_x: torch.Tensor
    ) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.canonicalize_value_basis()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
        for weight, qkv, proj_weight, _ in self.attention_specs:
            full_weight = qkv._initial_value_weight.to(
                device=weight.device, dtype=weight.dtype
            )
            delattr(qkv, "_initial_value_weight")
            full_shape = full_weight.shape
            width = full_shape[1]
            full_proj_shape = list(proj_weight.shape)
            full_proj_shape[0] += 1
            self.attention_states.append(
                {
                    "step": 0,
                    "full_weight": full_weight,
                    "scale": weight.new_ones(width),
=======
        for weight, qkv, proj_weight, _ in self.attention_specs:
            full_weight = qkv._initial_value_weight.to(
                device=weight.device, dtype=weight.dtype
            )
            virtual_proj_weight = qkv._initial_proj_weight.to(
                device=proj_weight.device, dtype=proj_weight.dtype
            )
            delattr(qkv, "_initial_value_weight")
            delattr(qkv, "_initial_proj_weight")
            full_shape = full_weight.shape
            width = full_shape[1]
            full_proj_shape = list(proj_weight.shape)
            full_proj_shape[0] += 1
            self.attention_states.append(
                {
                    "step": 0,
                    "full_weight": full_weight,
                    "proj_weight": virtual_proj_weight,
                    "scale": weight.new_ones(width),
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _full_attention_grads(
        weight_param, proj_weight_param, folded_bias_param, state
    ):
        stored_weight_grad = (
            torch.zeros_like(weight_param)
            if weight_param.grad is None
            else weight_param.grad.detach()
        )
        stored_proj_grad = (
            torch.zeros_like(proj_weight_param)
            if proj_weight_param.grad is None
            else proj_weight_param.grad.detach()
        )
        folded_grad = (
            torch.zeros_like(folded_bias_param)
            if folded_bias_param.grad is None
            else folded_bias_param.grad.detach()
        )

        effective_reference_grad = -stored_weight_grad.sum(
            dim=1, keepdim=True
        )
        full_effective_grad = torch.cat(
            [stored_weight_grad, effective_reference_grad], dim=1
        )

        constant = (
            state["full_weight"].mv(state["shift"])
            + state["full_bias"]
        )
        constant_grad = proj_weight_param.transpose(0, 1).mv(
            folded_grad
        )

        combined_proj_grad = stored_proj_grad + (
            folded_grad.unsqueeze(1) * constant.unsqueeze(0)
        )
        full_proj_grad = torch.cat(
            [
                combined_proj_grad,
                -combined_proj_grad.sum(dim=0, keepdim=True),
            ],
            dim=0,
        )

        full_weight_grad = (
            full_effective_grad * state["scale"].unsqueeze(0)
        )
        full_weight_grad = full_weight_grad + (
            constant_grad.unsqueeze(1)
            * state["shift"].unsqueeze(0)
        )
        scale_grad = (
            full_effective_grad * state["full_weight"]
        ).sum(dim=0)
        shift_grad = state["full_weight"].transpose(0, 1).mv(
            constant_grad
        )
        full_bias_grad = constant_grad + torch.cat(
            [folded_grad, -folded_grad.sum().reshape(1)]
        )
        return (
            full_weight_grad,
            scale_grad,
            shift_grad,
            full_proj_grad,
            full_bias_grad,
        )
=======
    @staticmethod
    def _full_attention_grads(
        weight_param, proj_weight_param, folded_bias_param, state
    ):
        canonical_tail_grad = (
            torch.zeros_like(weight_param)
            if weight_param.grad is None
            else weight_param.grad.detach()
        )
        canonical_proj_grad = (
            torch.zeros_like(proj_weight_param)
            if proj_weight_param.grad is None
            else proj_weight_param.grad.detach()
        )
        folded_grad = (
            torch.zeros_like(folded_bias_param)
            if folded_bias_param.grad is None
            else folded_bias_param.grad.detach()
        )

        n_head, head_dim, tail_width = weight_param.shape
        reduced_width = head_dim + tail_width
        scaled_weight = (
            state["full_weight"] * state["scale"].unsqueeze(0)
        )
        reduced_weight = (
            scaled_weight[:, :-1] - scaled_weight[:, -1:]
        )
        virtual_value_heads = reduced_weight.view(
            n_head, head_dim, reduced_width
        )
        value_basis = virtual_value_heads[..., :head_dim]

        canonical_proj_heads = proj_weight_param.view(
            proj_weight_param.shape[0], n_head, head_dim
        )
        canonical_proj_grad_heads = canonical_proj_grad.view(
            canonical_proj_grad.shape[0], n_head, head_dim
        )

        # Recover the gradient of the fixed identity value block from the
        # exact per-head value/output change-of-basis invariance.
        canonical_prefix_grad = torch.einsum(
            "rha,rhb->hab",
            canonical_proj_heads,
            canonical_proj_grad_heads,
        )
        canonical_prefix_grad = canonical_prefix_grad - torch.matmul(
            canonical_tail_grad,
            weight_param.transpose(-1, -2),
        )
        canonical_value_grad = torch.cat(
            [canonical_prefix_grad, canonical_tail_grad], dim=-1
        )

        # Map canonical-factor gradients back to the virtual value and
        # output factors whose original AdamW moments are retained.
        virtual_value_grad = torch.linalg.solve(
            value_basis.transpose(-1, -2),
            canonical_value_grad,
        ).reshape(n_head * head_dim, reduced_width)
        direct_proj_grad = torch.einsum(
            "rha,hba->rhb",
            canonical_proj_grad_heads,
            value_basis,
        ).reshape_as(proj_weight_param)

        effective_reference_grad = -virtual_value_grad.sum(
            dim=1, keepdim=True
        )
        full_effective_grad = torch.cat(
            [virtual_value_grad, effective_reference_grad], dim=1
        )

        constant = (
            state["full_weight"].mv(state["shift"])
            + state["full_bias"]
        )
        constant_grad = state["proj_weight"].transpose(0, 1).mv(
            folded_grad
        )

        combined_proj_grad = direct_proj_grad + (
            folded_grad.unsqueeze(1) * constant.unsqueeze(0)
        )
        full_proj_grad = torch.cat(
            [
                combined_proj_grad,
                -combined_proj_grad.sum(dim=0, keepdim=True),
            ],
            dim=0,
        )

        full_weight_grad = (
            full_effective_grad * state["scale"].unsqueeze(0)
        )
        full_weight_grad = full_weight_grad + (
            constant_grad.unsqueeze(1)
            * state["shift"].unsqueeze(0)
        )
        scale_grad = (
            full_effective_grad * state["full_weight"]
        ).sum(dim=0)
        shift_grad = state["full_weight"].transpose(0, 1).mv(
            constant_grad
        )
        full_bias_grad = constant_grad + torch.cat(
            [folded_grad, -folded_grad.sum().reshape(1)]
        )
        return (
            full_weight_grad,
            scale_grad,
            shift_grad,
            full_proj_grad,
            full_bias_grad,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            update_quotient(
                proj_weight,
                proj_weight_grad,
                state["exp_avg_proj"],
                state["exp_avg_sq_proj"],
                bias_correction1,
                bias_correction2,
                0,
            )

            scaled_weight = (
                state["full_weight"]
                * state["scale"].unsqueeze(0)
            )
            weight.copy_(
                scaled_weight[:, :-1] - scaled_weight[:, -1:]
            )
            constant = (
                state["full_weight"].mv(state["shift"])
                + state["full_bias"]
            )
            folded_bias.copy_(proj_weight.mv(constant))
            folded_bias.add_(
                state["full_bias"][:-1] - state["full_bias"][-1]
            )
=======
            update_quotient(
                state["proj_weight"],
                proj_weight_grad,
                state["exp_avg_proj"],
                state["exp_avg_sq_proj"],
                bias_correction1,
                bias_correction2,
                0,
            )

            scaled_weight = (
                state["full_weight"]
                * state["scale"].unsqueeze(0)
            )
            reduced_weight = (
                scaled_weight[:, :-1] - scaled_weight[:, -1:]
            )
            n_head, head_dim, tail_width = weight.shape
            reduced_width = head_dim + tail_width
            value_heads = reduced_weight.view(
                n_head, head_dim, reduced_width
            )
            value_basis = value_heads[..., :head_dim]
            weight.copy_(
                torch.linalg.solve(
                    value_basis,
                    value_heads[..., head_dim:],
                )
            )

            virtual_proj_heads = state["proj_weight"].view(
                state["proj_weight"].shape[0],
                n_head,
                head_dim,
            )
            canonical_proj = torch.einsum(
                "rhd,hde->rhe",
                virtual_proj_heads,
                value_basis,
            )
            proj_weight.copy_(
                canonical_proj.reshape_as(proj_weight)
            )

            constant = (
                state["full_weight"].mv(state["shift"])
                + state["full_bias"]
            )
            folded_bias.copy_(
                state["proj_weight"].mv(constant)
            )
            folded_bias.add_(
                state["full_bias"][:-1] - state["full_bias"][-1]
            )
>>>>>>> REPLACE