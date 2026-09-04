MECHANISM: Optimizer-preserving value/LayerNorm composite quotient

HYPOTHESIS: Collapsing the affine first LayerNorm and full value projection into 56 zero-mean-coordinate weights plus an 8-coordinate value constant will reduce the model from 1,422 to 1,406 parameters while maintaining at least 99% accuracy, because virtual full-width AdamW updates preserve the successful value-path optimization geometry.

INTENDED_EDIT: Replace the affine value pathway with its exact composite representation, retain the shared value/output bias, and train the compressed value parameters through virtual full value weights and LayerNorm scale/shift states.

EVIDENCE: Directly fixing one `ln1` scale coordinate fell to 74.8%, while optimizer-preserving query and key LayerNorm quotients reached 99.98% and 99.95%; this motivates quotienting the remaining value-only affine parameters without directly changing their AdamW dynamics.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Q/K projections modulo LayerNorm's all-ones direction plus full-width V."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model))

        # Preserve the RNG stream of the removed bias-free QKV Linear.
        discarded_weight = torch.empty(3 * d_model, d_model)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    def initialize_from_full_normal(self) -> None:
        full_weight = self.value_weight.new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        query_weight = full_weight[: self.d_model]
        key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
        with torch.no_grad():
            self.query_weight.copy_(
                query_weight[:, :-1] - query_weight[:, -1:]
            )
            self.key_weight.copy_(
                key_weight[:, :-1] - key_weight[:, -1:]
            )
            self.value_weight.copy_(
                full_weight[2 * self.d_model :]
            )

    def forward(
        self, affine_x: torch.Tensor, normalized_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.linear(normalized_x[..., :-1], self.query_weight)
        k = F.linear(normalized_x[..., :-1], self.key_weight)
        v = F.linear(affine_x, self.value_weight)
        return q, k, v
=======
class GaugeFixedQKV(nn.Module):
    """Q/K/V projections on LayerNorm's seven zero-mean coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_bias = nn.Parameter(torch.zeros(d_model))

        # Preserve the RNG stream of the removed bias-free QKV Linear.
        discarded_weight = torch.empty(3 * d_model, d_model)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

    def initialize_from_full_normal(self) -> None:
        full_weight = self.value_weight.new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(full_weight, mean=0.0, std=0.02)
        query_weight = full_weight[: self.d_model]
        key_weight = full_weight[
            self.d_model : 2 * self.d_model
        ]
        value_weight = full_weight[2 * self.d_model :]
        with torch.no_grad():
            self.query_weight.copy_(
                query_weight[:, :-1] - query_weight[:, -1:]
            )
            self.key_weight.copy_(
                key_weight[:, :-1] - key_weight[:, -1:]
            )
            self.value_weight.copy_(
                value_weight[:, :-1] - value_weight[:, -1:]
            )
            self.value_bias.zero_()

        # Consumed by QuotientAdamW to initialize the virtual full pathway.
        self._initial_value_weight = value_weight.detach().clone()

    def forward(
        self, affine_x: torch.Tensor, normalized_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = F.linear(normalized_x[..., :-1], self.query_weight)
        k = F.linear(normalized_x[..., :-1], self.key_weight)
        v = F.linear(
            normalized_x[..., :-1],
            self.value_weight,
            self.value_bias,
        )
        return q, k, v
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model, elementwise_affine=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        affine_normalized = (
            normalized * self.ln1.weight + self.ln1.bias
        )
        x = x + self.attn(affine_normalized, normalized)
=======
        x = x + self.attn(normalized, normalized)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def __init__(self, model: TinyDecoderLM, lr: float, weight_decay: float):
        self.all_params = list(model.parameters())
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
        self.gauge_params = [param for param, _ in self.gauge_specs]
        gauge_ids = {id(param) for param in self.gauge_params}
        ordinary_params = [
            param for param in self.all_params if id(param) not in gauge_ids
        ]

        self.base = torch.optim.AdamW(
            ordinary_params, lr=lr, weight_decay=weight_decay
        )
        self.param_groups = self.base.param_groups
        self.gauge_states = []
        for param, axis in self.gauge_specs:
            full_shape = list(param.shape)
            full_shape[axis] += 1
            self.gauge_states.append(
                {
                    "step": 0,
                    "exp_avg": param.new_zeros(full_shape),
                    "exp_avg_sq": param.new_zeros(full_shape),
                }
            )
=======
    def __init__(self, model: TinyDecoderLM, lr: float, weight_decay: float):
        self.all_params = list(model.parameters())
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

        self.base = torch.optim.AdamW(
            ordinary_params, lr=lr, weight_decay=weight_decay
        )
        self.param_groups = self.base.param_groups
        self.gauge_states = []
        for param, axis in self.gauge_specs:
            full_shape = list(param.shape)
            full_shape[axis] += 1
            self.gauge_states.append(
                {
                    "step": 0,
                    "exp_avg": param.new_zeros(full_shape),
                    "exp_avg_sq": param.new_zeros(full_shape),
                }
            )

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
>>>>>>> REPLACE

<<<<<<< SEARCH
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)
        for param in self.gauge_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
=======
    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)
        for param in self.gauge_params + self.value_params:
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

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
        for param in self.all_params:
            if param.grad is not None:
                grad = param.grad.detach().float()
                total_sq.add_(grad.square().sum())

        # Include each omitted reference coordinate's full-model gradient.
        for param, axis in self.gauge_specs:
            if param.grad is not None:
                omitted_grad = -param.grad.detach().sum(
                    dim=axis, keepdim=True
                )
                total_sq.add_(omitted_grad.float().square().sum())

        total_norm = total_sq.sqrt()
=======
        for param in self.all_params:
            if (
                id(param) not in self.value_param_ids
                and param.grad is not None
            ):
                grad = param.grad.detach().float()
                total_sq.add_(grad.square().sum())

        # Include each omitted reference coordinate's full-model gradient.
        for param, axis in self.gauge_specs:
            if param.grad is not None:
                omitted_grad = -param.grad.detach().sum(
                    dim=axis, keepdim=True
                )
                total_sq.add_(omitted_grad.float().square().sum())

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

        total_norm = total_sq.sqrt()
>>>>>>> REPLACE

<<<<<<< SEARCH
            param.add_(
                quotient_update, alpha=-lr / bias_correction1
            )


def save_json(path: Path, obj: Dict) -> None:
=======
            param.add_(
                quotient_update, alpha=-lr / bias_correction1
            )

        def update_virtual(
            value, grad, exp_avg, exp_avg_sq, bias_correction1,
            bias_correction2
        ):
            value.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            value.add_(
                exp_avg / denom,
                alpha=-lr / bias_correction1,
            )

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


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE