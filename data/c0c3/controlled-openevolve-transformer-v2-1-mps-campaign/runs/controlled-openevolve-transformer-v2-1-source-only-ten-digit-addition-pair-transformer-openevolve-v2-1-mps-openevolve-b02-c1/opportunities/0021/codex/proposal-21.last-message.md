MECHANISM: Optimizer-preserving attention-logit gauge quotient

HYPOTHESIS: Fixing one relative-attention bias per head at zero and training the remaining differences with virtual full-width AdamW moments will reduce the model from 1,449 to 1,447 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store 22 relative-distance bias differences per head instead of 23, reconstruct a fixed zero reference bias, and generalize the quotient-aware optimizer to preserve the omitted biases’ gradients and AdamW moments.

EVIDENCE: The 1,449-parameter relative-distance model achieved 99.99%, and quotient-aware gauge optimization previously preserved 99.75% accuracy when direct removal of functionally redundant MLP coordinates failed; this applies the same optimizer-preserving method to an exact per-head softmax-shift invariance.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len))
=======
        # One bias per head is a softmax-invisible common-mode reference.
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = att + self.rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
=======
        rel_bias = F.pad(self.rel_bias, (0, 1))
        att = att + rel_bias[:, distance.clamp_min(0)].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
class QuotientAdamW:
    """AdamW on seven stored differences with a virtual eighth output row."""

    def __init__(self, model: TinyDecoderLM, lr: float, weight_decay: float):
        self.all_params = list(model.parameters())
        self.gauge_params = [block.mlp.fc2.weight for block in model.blocks]
        gauge_ids = {id(param) for param in self.gauge_params}
        ordinary_params = [
            param for param in self.all_params if id(param) not in gauge_ids
        ]

        self.base = torch.optim.AdamW(
            ordinary_params, lr=lr, weight_decay=weight_decay
        )
        self.param_groups = self.base.param_groups
        self.gauge_states = [
            {
                "step": 0,
                "exp_avg": param.new_zeros(
                    param.shape[0] + 1, param.shape[1]
                ),
                "exp_avg_sq": param.new_zeros(
                    param.shape[0] + 1, param.shape[1]
                ),
            }
            for param in self.gauge_params
        ]
=======
class QuotientAdamW:
    """AdamW on stored gauge differences with one virtual reference coordinate."""

    def __init__(self, model: TinyDecoderLM, lr: float, weight_decay: float):
        self.all_params = list(model.parameters())
        self.gauge_specs = [
            (block.mlp.fc2.weight, 0) for block in model.blocks
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Include the omitted row's gradient in the norm used by the full model.
        for param in self.gauge_params:
            if param.grad is not None:
                omitted_grad = -param.grad.detach().sum(dim=0)
                total_sq.add_(omitted_grad.float().square().sum())
=======
        # Include each omitted reference coordinate's full-model gradient.
        for param, axis in self.gauge_specs:
            if param.grad is not None:
                omitted_grad = -param.grad.detach().sum(
                    dim=axis, keepdim=True
                )
                total_sq.add_(omitted_grad.float().square().sum())
>>>>>>> REPLACE

<<<<<<< SEARCH
        for param, state in zip(self.gauge_params, self.gauge_states):
            if param.grad is None:
                continue

            grad = param.grad
            full_grad = torch.cat(
                [grad, -grad.sum(dim=0, keepdim=True)], dim=0
            )
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            param.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            update = exp_avg / denom
            quotient_update = update[:-1] - update[-1:]
            param.add_(
                quotient_update, alpha=-lr / bias_correction1
            )
=======
        for (param, axis), state in zip(
            self.gauge_specs, self.gauge_states
        ):
            if param.grad is None:
                continue

            grad = param.grad
            full_grad = torch.cat(
                [grad, -grad.sum(dim=axis, keepdim=True)], dim=axis
            )
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            param.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            update = exp_avg / denom
            stored_update = update.narrow(
                axis, 0, param.shape[axis]
            )
            reference_update = update.narrow(
                axis, param.shape[axis], 1
            )
            quotient_update = stored_update - reference_update
            param.add_(
                quotient_update, alpha=-lr / bias_correction1
            )
>>>>>>> REPLACE