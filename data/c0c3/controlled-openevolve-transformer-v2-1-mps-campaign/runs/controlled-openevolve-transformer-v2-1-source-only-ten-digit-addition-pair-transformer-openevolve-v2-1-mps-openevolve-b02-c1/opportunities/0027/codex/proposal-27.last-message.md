MECHANISM: Optimizer-preserving coupled value/output-bias gauge quotient

HYPOTHESIS: Fixing one coordinate of the shared attention projection bias while compensating it in the value constant will reduce the model from 1,406 to 1,405 parameters and maintain at least 99% accuracy, because the transformation changes the residual only by a LayerNorm-invisible common shift and virtual full-width AdamW preserves the successful optimization geometry.

INTENDED_EDIT: Store seven projection-bias differences, reconstruct the eighth as zero, compensate its virtual reference coordinate in the value bias, and train both original full bias pathways with reconstructed AdamW gradients and moments.

EVIDENCE: The 1,406-parameter composite value/LayerNorm quotient achieved 99.99%, while direct single-coordinate gauge fixing previously failed; optimizer-preserving quotients repeatedly retained at least 99.75%, motivating this exact one-dimensional coupled quotient.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(out_features))
=======
        self.bias = nn.Parameter(torch.zeros(out_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 0, 0, 1))
        return F.linear(x, weight, self.bias)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 0, 0, 1))
        bias = F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = self.qkv(x, normalized_x)
        q = q + self.q_bias
        v = v + self.proj.bias
=======
        q, k, v = self.qkv(x, normalized_x)
        q = q + self.q_bias
        v = v + F.pad(self.proj.bias, (0, 1))
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
        self.gauge_params = [param for param, _ in self.gauge_specs]
        self.value_params = [
            param
            for weight, bias, _ in self.value_specs
            for param in (weight, bias)
        ]
        excluded_ids = {
            id(param) for param in self.gauge_params + self.value_params
        }
        self.value_param_ids = {
            id(param) for param in self.value_params
        }
        ordinary_params = [
            param
            for param in self.all_params
            if id(param) not in excluded_ids
        ]
=======
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
        ordinary_params = [
            param
            for param in self.all_params
            if id(param) not in excluded_ids
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        for param in self.gauge_params + self.value_params:
=======
        for param in (
            self.gauge_params
            + self.value_params
            + self.proj_bias_params
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
        return full_weight_grad, scale_grad, shift_grad

    @torch.no_grad()
=======
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

    @torch.no_grad()
>>>>>>> REPLACE

<<<<<<< SEARCH
                id(param) not in self.value_param_ids
=======
                id(param) not in self.custom_param_ids
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (weight, bias, _), state in zip(
            self.value_specs, self.value_states
        ):
            full_grads = self._full_value_grads(
                weight, bias, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())

        total_norm = total_sq.sqrt()
=======
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

        total_norm = total_sq.sqrt()
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (weight, bias, _), state in zip(
            self.value_specs, self.value_states
        ):
            if weight.grad is None and bias.grad is None:
                continue

            weight_grad, scale_grad, shift_grad = (
                self._full_value_grads(weight, bias, state)
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

            scaled_weight = (
                state["full_weight"]
                * state["scale"].unsqueeze(0)
            )
            weight.copy_(
                scaled_weight[:, :-1] - scaled_weight[:, -1:]
            )
            bias.copy_(
                state["full_weight"].mv(state["shift"])
            )
=======
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
>>>>>>> REPLACE