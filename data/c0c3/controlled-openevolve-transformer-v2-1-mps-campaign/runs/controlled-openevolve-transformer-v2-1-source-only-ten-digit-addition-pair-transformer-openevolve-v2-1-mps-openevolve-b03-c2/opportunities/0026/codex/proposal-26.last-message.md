MECHANISM: Multi-coordinate value-bias absorption into the attention projection bias

HYPOTHESIS: Extending the qualified value-bias quotient from one to two omitted coordinates will produce a 1593-parameter model with at least 99% accuracy while preserving both omitted coordinates’ clipping and AdamW dynamics.

INTENDED_EDIT: Store six value-bias coordinates, reconstruct two zero-gauge coordinates, and generalize the optimizer and gradient clipping to track and absorb both omitted updates.

EVIDENCE: The current 1594-parameter design achieved 99.73% after one value-bias coordinate was absorbed into `attn.proj.bias`; the second coordinate has the same attention-invariant computational role.

<<<<<<< SEARCH
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain seven value-bias coordinates; the omitted eighth value
        # coordinate is absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 1))
=======
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but two value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(1))
        )
=======
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(2))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias_grads = []
        for _, proj_weight, proj_bias in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                grad = (
                    proj_weight.detach()[:, -1]
                    * proj_bias.grad.detach()
                ).sum()
            value_bias_grads.append(grad)
=======
        value_bias_grads = []
        for qkv_bias, proj_weight, proj_bias in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                omitted_dims = (
                    2 * proj_weight.size(1) - qkv_bias.numel()
                )
                grad = (
                    proj_weight.detach()[:, -omitted_dims:]
                    * proj_bias.grad.detach().unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)
>>>>>>> REPLACE

<<<<<<< SEARCH
            if "value_quotient_step" not in state:
                state["value_quotient_step"] = 0
                state["value_quotient_exp_avg"] = qkv_bias.new_zeros(())
                state["value_quotient_exp_avg_sq"] = qkv_bias.new_zeros(())
=======
            if "value_quotient_step" not in state:
                state["value_quotient_step"] = 0
                state["value_quotient_exp_avg"] = qkv_bias.new_zeros(
                    grad.shape
                )
                state["value_quotient_exp_avg_sq"] = qkv_bias.new_zeros(
                    grad.shape
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
            omitted_value = -step_size * exp_avg / denom
            proj_bias.add_(proj_weight[:, -1] * omitted_value)
=======
            omitted_value = -step_size * exp_avg / denom
            proj_bias.add_(
                proj_weight[:, -omitted_value.numel():] @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for _, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_grad = (
            proj_weight.detach()[:, -1] * proj_bias.grad.detach()
        ).sum()
        term = omitted_grad.square()
        total_sq = term if total_sq is None else total_sq + term
=======
    for qkv_bias, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_dims = 2 * proj_weight.size(1) - qkv_bias.numel()
        omitted_grad = (
            proj_weight.detach()[:, -omitted_dims:]
            * proj_bias.grad.detach().unsqueeze(1)
        ).sum(dim=0)
        term = omitted_grad.square().sum()
        total_sq = term if total_sq is None else total_sq + term
>>>>>>> REPLACE