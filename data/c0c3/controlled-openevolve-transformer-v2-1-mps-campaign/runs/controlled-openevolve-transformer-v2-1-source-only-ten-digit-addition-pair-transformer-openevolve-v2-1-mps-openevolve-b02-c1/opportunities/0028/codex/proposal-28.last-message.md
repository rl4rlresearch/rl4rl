MECHANISM: Optimizer-preserving attention constant folding

HYPOTHESIS: Folding the eight-coordinate value/output constant into the seven observable residual-bias coordinates will reduce the model from 1,405 to 1,397 parameters while retaining at least 99% accuracy, because virtual full-width AdamW gradients preserve the successful optimization trajectory.

INTENDED_EDIT: Remove the learned QKV value bias, use the residual projection bias as the folded effective constant, and reconstruct gradients and updates for the original value/LayerNorm and projection-bias pathways inside the quotient optimizer.

EVIDENCE: The 1,405-parameter coupled value/output-bias quotient achieved 99.97%, and prior optimizer-preserving quotients consistently retained high accuracy. In that design the value constant and full projection bias affect the residual only through `W_out(c + p) + p`, which has seven observable coordinates.

<<<<<<< SEARCH
        self.query_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.query_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = F.linear(
            normalized_x[..., :-1],
            self.value_weight,
            self.value_bias,
        )
=======
        v = F.linear(
            normalized_x[..., :-1],
            self.value_weight,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = self.qkv(x, normalized_x)
        q = q + self.q_bias
        v = v + F.pad(self.proj.bias, (0, 1))

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = self.qkv(x, normalized_x)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value_specs = [
            (
                block.attn.qkv.value_weight,
                block.attn.qkv.value_bias,
                block.attn.qkv,
            )
            for block in model.blocks
        ]
        self.proj_bias_specs = [
            (
                block.attn.proj.bias,
                block.attn.qkv.value_bias,
            )
            for block in model.blocks
        ]
        self.gauge_params = [param for param, _ in self.gauge_specs]
        self.value_params = [
            param
            for weight, bias, _ in self.value_specs
            for param in (weight, bias)
        ]
        self.proj_bias_params = [
            proj_bias for proj_bias, _ in self.proj_bias_specs
        ]
=======
        self.value_specs = [
            (
                block.attn.qkv.value_weight,
                block.attn.qkv,
            )
            for block in model.blocks
        ]
        self.proj_bias_specs = [
            (
                block.attn.proj.weight,
                block.attn.proj.bias,
            )
            for block in model.blocks
        ]
        self.gauge_params = [param for param, _ in self.gauge_specs]
        self.value_params = [
            weight for weight, _ in self.value_specs
        ]
        self.proj_bias_params = [
            effective_bias
            for _, effective_bias in self.proj_bias_specs
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value_states = []
        for weight, _, qkv in self.value_specs:
            full_weight = qkv._initial_value_weight.to(
                device=weight.device, dtype=weight.dtype
            )
            delattr(qkv, "_initial_value_weight")
            full_shape = full_weight.shape
            width = full_shape[1]
            self.value_states.append(
                {
                    "step": 0,
                    "full_weight": full_weight,
                    "scale": weight.new_ones(width),
                    "shift": weight.new_zeros(width),
                    "exp_avg_weight": weight.new_zeros(full_shape),
                    "exp_avg_sq_weight": weight.new_zeros(full_shape),
                    "exp_avg_scale": weight.new_zeros(width),
                    "exp_avg_sq_scale": weight.new_zeros(width),
                    "exp_avg_shift": weight.new_zeros(width),
                    "exp_avg_sq_shift": weight.new_zeros(width),
                }
            )

        self.proj_bias_states = []
        for proj_bias, _ in self.proj_bias_specs:
            full_bias = torch.cat(
                [proj_bias.detach().clone(), proj_bias.new_zeros(1)]
            )
            self.proj_bias_states.append(
                {
                    "step": 0,
                    "full_bias": full_bias,
                    "exp_avg": torch.zeros_like(full_bias),
                    "exp_avg_sq": torch.zeros_like(full_bias),
                }
            )
=======
        self.value_states = []
        for weight, qkv in self.value_specs:
            full_weight = qkv._initial_value_weight.to(
                device=weight.device, dtype=weight.dtype
            )
            delattr(qkv, "_initial_value_weight")
            full_shape = full_weight.shape
            width = full_shape[1]
            self.value_states.append(
                {
                    "step": 0,
                    "full_weight": full_weight,
                    "scale": weight.new_ones(width),
                    "shift": weight.new_zeros(width),
                    "exp_avg_weight": weight.new_zeros(full_shape),
                    "exp_avg_sq_weight": weight.new_zeros(full_shape),
                    "exp_avg_scale": weight.new_zeros(width),
                    "exp_avg_sq_scale": weight.new_zeros(width),
                    "exp_avg_shift": weight.new_zeros(width),
                    "exp_avg_sq_shift": weight.new_zeros(width),
                }
            )

        self.proj_bias_states = []
        for _, effective_bias in self.proj_bias_specs:
            full_bias = torch.cat(
                [
                    effective_bias.detach().clone(),
                    effective_bias.new_zeros(1),
                ]
            )
            self.proj_bias_states.append(
                {
                    "step": 0,
                    "full_bias": full_bias,
                    "exp_avg": torch.zeros_like(full_bias),
                    "exp_avg_sq": torch.zeros_like(full_bias),
                }
            )
        self._attention_grads_prepared = False
>>>>>>> REPLACE

<<<<<<< SEARCH
        for param in (
            self.gauge_params
            + self.value_params
            + self.proj_bias_params
        ):
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    @staticmethod
    def _full_value_grads(weight_param, bias_param, state):
        stored_weight_grad = (
            torch.zeros_like(weight_param)
            if weight_param.grad is None
            else weight_param.grad.detach()
        )
        stored_bias_grad = (
            torch.zeros_like(bias_param)
            if bias_param.grad is None
            else bias_param.grad.detach()
        )
        reference_grad = -stored_weight_grad.sum(
            dim=1, keepdim=True
        )

        full_weight_grad = torch.cat(
            [
                stored_weight_grad
                * state["scale"][:-1].unsqueeze(0),
                reference_grad * state["scale"][-1],
            ],
            dim=1,
        )
        full_weight_grad = full_weight_grad + (
            stored_bias_grad.unsqueeze(1)
            * state["shift"].unsqueeze(0)
        )

        scale_grad = torch.cat(
            [
                (
                    stored_weight_grad
                    * state["full_weight"][:, :-1]
                ).sum(dim=0),
                (
                    reference_grad
                    * state["full_weight"][:, -1:]
                ).sum().reshape(1),
            ]
        )
        shift_grad = state["full_weight"].transpose(0, 1).mv(
            stored_bias_grad
        )
        return full_weight_grad, scale_grad, shift_grad

    @staticmethod
    def _full_proj_bias_grad(proj_bias_param, value_bias_param):
        stored_proj_grad = (
            torch.zeros_like(proj_bias_param)
            if proj_bias_param.grad is None
            else proj_bias_param.grad.detach()
        )
        value_bias_grad = (
            torch.zeros_like(value_bias_param)
            if value_bias_param.grad is None
            else value_bias_param.grad.detach()
        )
        reference_grad = (
            value_bias_grad.sum() - stored_proj_grad.sum()
        ).reshape(1)
        return torch.cat([stored_proj_grad, reference_grad])
=======
        for param in (
            self.gauge_params
            + self.value_params
            + self.proj_bias_params
        ):
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()
        self._attention_grads_prepared = False

    @staticmethod
    def _full_value_grads(weight_param, value_bias_grad, state):
        stored_weight_grad = (
            torch.zeros_like(weight_param)
            if weight_param.grad is None
            else weight_param.grad.detach()
        )
        stored_bias_grad = value_bias_grad.detach()
        reference_grad = -stored_weight_grad.sum(
            dim=1, keepdim=True
        )

        full_weight_grad = torch.cat(
            [
                stored_weight_grad
                * state["scale"][:-1].unsqueeze(0),
                reference_grad * state["scale"][-1],
            ],
            dim=1,
        )
        full_weight_grad = full_weight_grad + (
            stored_bias_grad.unsqueeze(1)
            * state["shift"].unsqueeze(0)
        )

        scale_grad = torch.cat(
            [
                (
                    stored_weight_grad
                    * state["full_weight"][:, :-1]
                ).sum(dim=0),
                (
                    reference_grad
                    * state["full_weight"][:, -1:]
                ).sum().reshape(1),
            ]
        )
        shift_grad = state["full_weight"].transpose(0, 1).mv(
            stored_bias_grad
        )
        return full_weight_grad, scale_grad, shift_grad

    @staticmethod
    def _attention_constant_grads(
        proj_weight, effective_bias, value_state, proj_state
    ):
        effective_grad = (
            torch.zeros_like(effective_bias)
            if effective_bias.grad is None
            else effective_bias.grad.detach()
        )
        value_constant = value_state["full_weight"].mv(
            value_state["shift"]
        )
        full_constant = value_constant + proj_state["full_bias"]

        value_bias_grad = proj_weight.transpose(0, 1).mv(
            effective_grad
        )
        direct_proj_grad = torch.cat(
            [effective_grad, -effective_grad.sum().reshape(1)]
        )
        full_proj_bias_grad = value_bias_grad + direct_proj_grad
        proj_weight_extra_grad = (
            effective_grad.unsqueeze(1)
            * full_constant.unsqueeze(0)
        )
        return (
            value_bias_grad,
            full_proj_bias_grad,
            proj_weight_extra_grad,
        )

    @torch.no_grad()
    def _prepare_attention_grads(self) -> None:
        if self._attention_grads_prepared:
            return
        for (
            (proj_weight, effective_bias),
            value_state,
            proj_state,
        ) in zip(
            self.proj_bias_specs,
            self.value_states,
            self.proj_bias_states,
        ):
            if effective_bias.grad is None:
                continue
            _, _, extra_grad = self._attention_constant_grads(
                proj_weight,
                effective_bias,
                value_state,
                proj_state,
            )
            if proj_weight.grad is None:
                proj_weight.grad = extra_grad.clone()
            else:
                proj_weight.grad.add_(extra_grad)
        self._attention_grads_prepared = True
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        device = self.gauge_params[0].device
=======
    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        self._prepare_attention_grads()
        device = self.gauge_params[0].device
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Replace compressed value gradients with the gradients of the
        # virtual full value projection and affine LayerNorm parameters.
        for (weight, bias, _), state in zip(
            self.value_specs, self.value_states
        ):
            full_grads = self._full_value_grads(
                weight, bias, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())

        for proj_bias, value_bias in self.proj_bias_specs:
            grad = self._full_proj_bias_grad(
                proj_bias, value_bias
            )
            total_sq.add_(grad.float().square().sum())
=======
        # Replace compressed constant gradients with gradients of the
        # virtual value/LayerNorm and full projection-bias pathways.
        for (
            (weight, _),
            state,
            (proj_weight, effective_bias),
            proj_state,
        ) in zip(
            self.value_specs,
            self.value_states,
            self.proj_bias_specs,
            self.proj_bias_states,
        ):
            value_bias_grad, proj_bias_grad, _ = (
                self._attention_constant_grads(
                    proj_weight,
                    effective_bias,
                    state,
                    proj_state,
                )
            )
            full_grads = self._full_value_grads(
                weight, value_bias_grad, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())
            total_sq.add_(proj_bias_grad.float().square().sum())
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def step(self) -> None:
        self.base.step()

        group = self.param_groups[0]
=======
    @torch.no_grad()
    def step(self) -> None:
        self._prepare_attention_grads()
        constant_grads = [
            self._attention_constant_grads(
                proj_weight,
                effective_bias,
                value_state,
                proj_state,
            )
            for (
                (proj_weight, effective_bias),
                value_state,
                proj_state,
            ) in zip(
                self.proj_bias_specs,
                self.value_states,
                self.proj_bias_states,
            )
        ]
        self.base.step()

        group = self.param_groups[0]
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            (weight, bias, _),
            state,
            (proj_bias, _),
            proj_state,
        ) in zip(
            self.value_specs,
            self.value_states,
            self.proj_bias_specs,
            self.proj_bias_states,
        ):
            if (
                weight.grad is None
                and bias.grad is None
                and proj_bias.grad is None
            ):
                continue

            weight_grad, scale_grad, shift_grad = (
                self._full_value_grads(weight, bias, state)
            )
            proj_bias_grad = self._full_proj_bias_grad(
                proj_bias, bias
            )
=======
        for (
            (weight, _),
            state,
            (proj_weight, effective_bias),
            proj_state,
            (
                value_bias_grad,
                proj_bias_grad,
                _,
            ),
        ) in zip(
            self.value_specs,
            self.value_states,
            self.proj_bias_specs,
            self.proj_bias_states,
            constant_grads,
        ):
            if (
                weight.grad is None
                and effective_bias.grad is None
            ):
                continue

            weight_grad, scale_grad, shift_grad = (
                self._full_value_grads(
                    weight, value_bias_grad, state
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            scaled_weight = (
                state["full_weight"]
                * state["scale"].unsqueeze(0)
            )
            weight.copy_(
                scaled_weight[:, :-1] - scaled_weight[:, -1:]
            )
            reference = proj_state["full_bias"][-1]
            proj_bias.copy_(
                proj_state["full_bias"][:-1] - reference
            )
            bias.copy_(
                state["full_weight"].mv(state["shift"]) + reference
            )
=======
            scaled_weight = (
                state["full_weight"]
                * state["scale"].unsqueeze(0)
            )
            weight.copy_(
                scaled_weight[:, :-1] - scaled_weight[:, -1:]
            )
            value_constant = state["full_weight"].mv(
                state["shift"]
            )
            full_constant = (
                value_constant + proj_state["full_bias"]
            )
            reference = proj_state["full_bias"][-1]
            effective_bias.copy_(
                proj_weight.mv(full_constant)
                + proj_state["full_bias"][:-1]
                - reference
            )
>>>>>>> REPLACE