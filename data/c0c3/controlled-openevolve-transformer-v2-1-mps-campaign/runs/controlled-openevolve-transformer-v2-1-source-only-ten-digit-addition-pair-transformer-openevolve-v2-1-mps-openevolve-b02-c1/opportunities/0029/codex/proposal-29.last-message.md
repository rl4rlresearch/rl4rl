MECHANISM: Optimizer-preserving attention constant folding with joint chain-rule gradients

HYPOTHESIS: Folding the value/output constant into the seven observable residual-bias coordinates will produce a 1,397-parameter model with at least 99% accuracy when gradients for the virtual value, LayerNorm, projection-weight, and shared-bias parameters are reconstructed jointly.

INTENDED_EDIT: Remove the eight-parameter value bias, store only the folded residual constant, and extend quotient-aware AdamW to reconstruct and update every virtual parameter contributing to that constant.

EVIDENCE: The 1,405-parameter coupled value/output-bias quotient achieved 99.97%; the previous 1,397 constant-folding implementation could not be verified and therefore provides no accuracy counterevidence. Completing the joint chain rule preserves the successful model’s virtual optimization geometry.

<<<<<<< SEARCH
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.value_weight.copy_(
                value_weight[:, :-1] - value_weight[:, -1:]
            )
            self.value_bias.zero_()
=======
            self.value_weight.copy_(
                value_weight[:, :-1] - value_weight[:, -1:]
            )
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
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
            (block.attn.qkv.query_weight, 1) for block in model.blocks
        ] + [
            (block.attn.qkv.key_weight, 1) for block in model.blocks
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.proj.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
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
        excluded_ids = {
            id(param)
            for param in (
                self.gauge_params
                + self.value_params
                + self.proj_bias_params
            )
        }
        self.custom_param_ids = {
            id(param)
            for param in self.value_params + self.proj_bias_params
        }
=======
        self.gauge_specs = [
            (model.token_emb.weight, 0)
        ] + [
            (block.attn.qkv.query_weight, 1) for block in model.blocks
        ] + [
            (block.attn.qkv.key_weight, 1) for block in model.blocks
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
        self.attention_specs = [
            (
                block.attn.qkv.value_weight,
                block.attn.qkv,
                block.attn.proj.weight,
                block.attn.proj.bias,
            )
            for block in model.blocks
        ]
        self.gauge_params = [param for param, _ in self.gauge_specs]
        self.attention_params = [
            param
            for weight, _, proj_weight, folded_bias
            in self.attention_specs
            for param in (weight, proj_weight, folded_bias)
        ]
        excluded_ids = {
            id(param)
            for param in self.gauge_params + self.attention_params
        }
        self.custom_param_ids = {
            id(param) for param in self.attention_params
        }
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        for param in (
            self.gauge_params
            + self.value_params
            + self.proj_bias_params
        ):
=======
        for param in self.gauge_params + self.attention_params:
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        # Replace folded attention gradients with gradients of the virtual
        # value, LayerNorm, full projection, and shared-bias parameters.
        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
            self.attention_specs, self.attention_states
        ):
            full_grads = self._full_attention_grads(
                weight, proj_weight, folded_bias, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())
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

            state["step"] += 1
            step = state["step"]
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            update_virtual(
                state["full_weight"],
                weight_grad,
                state["exp_avg_weight"],
                state["exp_avg_sq_weight"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["scale"],
                scale_grad,
                state["exp_avg_scale"],
                state["exp_avg_sq_scale"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["shift"],
                shift_grad,
                state["exp_avg_shift"],
                state["exp_avg_sq_shift"],
                bias_correction1,
                bias_correction2,
            )

            proj_state["step"] += 1
            proj_step = proj_state["step"]
            proj_bias_correction1 = 1.0 - beta1 ** proj_step
            proj_bias_correction2 = 1.0 - beta2 ** proj_step
            update_virtual(
                proj_state["full_bias"],
                proj_bias_grad,
                proj_state["exp_avg"],
                proj_state["exp_avg_sq"],
                proj_bias_correction1,
                proj_bias_correction2,
            )

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
        def update_quotient(
            value, full_grad, exp_avg, exp_avg_sq,
            bias_correction1, bias_correction2
        ):
            value.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            update = exp_avg / denom
            stored_update = update[:-1]
            reference_update = update[-1:]
            value.add_(
                stored_update - reference_update,
                alpha=-lr / bias_correction1,
            )

        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
            self.attention_specs, self.attention_states
        ):
            if (
                weight.grad is None
                and proj_weight.grad is None
                and folded_bias.grad is None
            ):
                continue

            (
                weight_grad,
                scale_grad,
                shift_grad,
                proj_weight_grad,
                full_bias_grad,
            ) = self._full_attention_grads(
                weight, proj_weight, folded_bias, state
            )

            state["step"] += 1
            step = state["step"]
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            update_virtual(
                state["full_weight"],
                weight_grad,
                state["exp_avg_weight"],
                state["exp_avg_sq_weight"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["scale"],
                scale_grad,
                state["exp_avg_scale"],
                state["exp_avg_sq_scale"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["shift"],
                shift_grad,
                state["exp_avg_shift"],
                state["exp_avg_sq_shift"],
                bias_correction1,
                bias_correction2,
            )
            update_virtual(
                state["full_bias"],
                full_bias_grad,
                state["exp_avg_bias"],
                state["exp_avg_sq_bias"],
                bias_correction1,
                bias_correction2,
            )
            update_quotient(
                proj_weight,
                proj_weight_grad,
                state["exp_avg_proj"],
                state["exp_avg_sq_proj"],
                bias_correction1,
                bias_correction2,
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
>>>>>>> REPLACE