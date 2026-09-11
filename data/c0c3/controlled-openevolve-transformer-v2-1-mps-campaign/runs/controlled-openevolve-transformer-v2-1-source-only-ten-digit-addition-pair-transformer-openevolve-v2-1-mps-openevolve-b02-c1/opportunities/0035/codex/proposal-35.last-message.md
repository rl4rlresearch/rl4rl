MECHANISM: Optimizer-preserving per-head value/output basis gauge quotient

HYPOTHESIS: Canonicalizing each head’s value projection while retaining the pre-canonical projection differences and reconstructing their joint virtual gradients will reduce the model from 1,251 to 1,219 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Store only each value head’s three-column canonical tail, absorb its learned four-dimensional basis into the corresponding output-projection columns, and update the original virtual value, LayerNorm, projection, and bias parameters before recanonicalizing after every optimizer step.

EVIDENCE: Q/K basis canonicalization removed 24 parameters and achieved 99.98%, while joint attention folding achieved 99.93%. The earlier 1,219-parameter value/output attempt could not be verified and therefore supplies no accuracy counterevidence; this patch explicitly retains the virtual pre-canonical projection state needed for the coupled chain rule.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Basis-gauge-fixed Q/K factors and a zero-mean value projection."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        reduced_width = d_model - 1
        self.query_weight = nn.Parameter(
            torch.empty(n_head, self.head_dim, reduced_width)
        )
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.head_dim,
                reduced_width - self.head_dim,
            )
        )
        self.query_bias = nn.Parameter(
            torch.empty(n_head, self.head_dim)
        )
        self.value_weight = nn.Parameter(torch.empty(d_model, reduced_width))
=======
class GaugeFixedQKV(nn.Module):
    """Basis-gauge-fixed Q/K and value factors."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        reduced_width = d_model - 1
        self.query_weight = nn.Parameter(
            torch.empty(n_head, self.head_dim, reduced_width)
        )
        self.key_tail = nn.Parameter(
            torch.empty(
                n_head,
                self.head_dim,
                reduced_width - self.head_dim,
            )
        )
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
    def initialize_from_full_normal(self) -> None:
        full_weight = self.value_weight.new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        full_query_weight = full_weight[: self.d_model]
        full_key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
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
    def initialize_from_full_normal(self) -> None:
        full_weight = self.value_tail.new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        full_query_weight = full_weight[: self.d_model]
        full_key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
        value_weight = full_weight[2 * self.d_model :]
        query_weight = (
            full_query_weight[:, :-1] - full_query_weight[:, -1:]
        )
        key_weight = (
            full_key_weight[:, :-1] - full_key_weight[:, -1:]
        )
        effective_value_weight = (
            value_weight[:, :-1] - value_weight[:, -1:]
        )
        query_bias = full_weight.new_zeros(self.d_model)
        query_heads = query_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        key_heads = key_weight.view(
            self.n_head, self.head_dim, self.d_model - 1
        )
        value_heads = effective_value_weight.view(
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
                    value_basis, value_heads[..., self.head_dim :]
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
    def initialize_value_basis(self) -> None:
        full_value_weight = self.qkv._initial_value_weight
        effective_value_weight = (
            full_value_weight[:, :-1] - full_value_weight[:, -1:]
        )
        value_heads = effective_value_weight.view(
            self.n_head, self.head_dim, -1
        )
        value_basis = value_heads[..., : self.head_dim]
        self.qkv.value_tail.copy_(
            torch.linalg.solve(
                value_basis, value_heads[..., self.head_dim :]
            )
        )

        virtual_proj_weight = self.proj.weight.detach().clone()
        self.qkv._initial_proj_weight = virtual_proj_weight
        proj_heads = virtual_proj_weight.view(
            self.proj.out_features - 1,
            self.n_head,
            self.head_dim,
        )
        canonical_proj = torch.einsum(
            "ohd,hde->ohe", proj_heads, value_basis
        )
        self.proj.weight.copy_(
            canonical_proj.reshape_as(self.proj.weight)
        )

    def forward(
        self, x: torch.Tensor, normalized_x: torch.Tensor
    ) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.initialize_value_basis()
>>>>>>> REPLACE

<<<<<<< SEARCH
                block.attn.qkv.value_weight,
=======
                block.attn.qkv.value_tail,
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.attention_states = []
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
                    "shift": weight.new_zeros(width),
                    "full_bias": weight.new_zeros(width),
                    "exp_avg_weight": weight.new_zeros(full_shape),
                    "exp_avg_sq_weight": weight.new_zeros(full_shape),
                    "exp_avg_scale": weight.new_zeros(width),
                    "exp_avg_sq_scale": weight.new_zeros(width),
                    "exp_avg_shift": weight.new_zeros(width),
                    "exp_avg_sq_shift": weight.new_zeros(width),
                    "exp_avg_bias": weight.new_zeros(width),
                    "exp_avg_sq_bias": weight.new_zeros(width),
                    "exp_avg_proj": proj_weight.new_zeros(
                        full_proj_shape
                    ),
                    "exp_avg_sq_proj": proj_weight.new_zeros(
                        full_proj_shape
                    ),
                }
            )
=======
        self.attention_states = []
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
            full_proj_shape = list(virtual_proj_weight.shape)
            full_proj_shape[0] += 1
            self.attention_states.append(
                {
                    "step": 0,
                    "full_weight": full_weight,
                    "proj_weight": virtual_proj_weight,
                    "scale": weight.new_ones(width),
                    "shift": weight.new_zeros(width),
                    "full_bias": weight.new_zeros(width),
                    "exp_avg_weight": weight.new_zeros(full_shape),
                    "exp_avg_sq_weight": weight.new_zeros(full_shape),
                    "exp_avg_scale": weight.new_zeros(width),
                    "exp_avg_sq_scale": weight.new_zeros(width),
                    "exp_avg_shift": weight.new_zeros(width),
                    "exp_avg_sq_shift": weight.new_zeros(width),
                    "exp_avg_bias": weight.new_zeros(width),
                    "exp_avg_sq_bias": weight.new_zeros(width),
                    "exp_avg_proj": proj_weight.new_zeros(
                        full_proj_shape
                    ),
                    "exp_avg_sq_proj": proj_weight.new_zeros(
                        full_proj_shape
                    ),
                }
            )
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
        effective_weight = (
            scaled_weight[:, :-1] - scaled_weight[:, -1:]
        )
        value_heads = effective_weight.view(
            n_head, head_dim, reduced_width
        )
        value_basis = value_heads[..., :head_dim]

        virtual_proj = state["proj_weight"]
        proj_heads = virtual_proj.view(
            virtual_proj.shape[0], n_head, head_dim
        )
        canonical_proj_grad_heads = canonical_proj_grad.view(
            virtual_proj.shape[0], n_head, head_dim
        )

        # Map the learned canonical projection back to the virtual
        # projection and recover the fixed value-basis block's gradient.
        virtual_proj_grad_heads = torch.einsum(
            "ohe,hde->ohd",
            canonical_proj_grad_heads,
            value_basis,
        )
        solved_tail_grad = torch.linalg.solve(
            value_basis.transpose(-1, -2),
            canonical_tail_grad,
        )
        value_prefix_grad = torch.einsum(
            "ohd,ohe->hde",
            proj_heads,
            canonical_proj_grad_heads,
        )
        value_prefix_grad = value_prefix_grad - torch.matmul(
            solved_tail_grad,
            weight_param.transpose(-1, -2),
        )
        effective_weight_grad = torch.cat(
            [value_prefix_grad, solved_tail_grad], dim=-1
        ).reshape_as(effective_weight)

        effective_reference_grad = -effective_weight_grad.sum(
            dim=1, keepdim=True
        )
        full_effective_grad = torch.cat(
            [effective_weight_grad, effective_reference_grad],
            dim=1,
        )

        constant = (
            state["full_weight"].mv(state["shift"])
            + state["full_bias"]
        )
        constant_grad = virtual_proj.transpose(0, 1).mv(
            folded_grad
        )

        virtual_proj_grad = virtual_proj_grad_heads.reshape_as(
            virtual_proj
        )
        virtual_proj_grad = virtual_proj_grad + (
            folded_grad.unsqueeze(1) * constant.unsqueeze(0)
        )
        full_proj_grad = torch.cat(
            [
                virtual_proj_grad,
                -virtual_proj_grad.sum(dim=0, keepdim=True),
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
            effective_weight = (
                scaled_weight[:, :-1] - scaled_weight[:, -1:]
            )
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

            virtual_proj_heads = state["proj_weight"].view(
                proj_weight.shape[0], n_head, head_dim
            )
            canonical_proj = torch.einsum(
                "ohd,hde->ohe",
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