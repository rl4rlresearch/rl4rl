MECHANISM: Retained-coordinate rotation in the value-bias quotient

HYPOTHESIS: Retaining value-bias coordinate 1 while absorbing coordinates 0 and 2–7 will produce a 1588-parameter model with at least 99% accuracy, showing whether the failed prior reduction was caused by discarding coordinate 1 rather than by the seven-coordinate quotient itself.

INTENDED_EDIT: Store only the second value-bias coordinate, reconstruct the other seven as zero, and generalize quotient-aware clipping and AdamW absorption to noncontiguous omitted coordinates.

EVIDENCE: The 1589-parameter model retaining value coordinates 0 and 1 achieved 99.73%, whereas the 1588-parameter model retaining only coordinate 0 collapsed to 73.26%; retaining coordinate 1 instead directly tests the coordinate implicated by that regression.

<<<<<<< SEARCH
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain all but six value-bias coordinates; the omitted values
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 6))
=======
        # Key bias is softmax-invariant. Reuse the query-bias mean for it,
        # and retain only the second value-bias coordinate; the other seven
        # are absorbed by the downstream projection bias.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = torch.cat(
            (self.qkv.bias[d_model:], self.qkv.bias.new_zeros(6))
        )
=======
        retained_value_bias = self.qkv.bias[d_model:]
        value_bias = torch.cat(
            (
                retained_value_bias.new_zeros(1),
                retained_value_bias,
                retained_value_bias.new_zeros(d_model - 2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        for (
            qkv_bias,
            proj_weight,
            proj_bias,
            retained_dims,
        ) in self.value_bias_specs:
            grad = None
            if proj_bias.grad is not None:
                omitted_dims = [
                    dim
                    for dim in range(proj_weight.size(1))
                    if dim not in retained_dims
                ]
                grad = (
                    proj_weight.detach()[:, omitted_dims]
                    * proj_bias.grad.detach().unsqueeze(1)
                ).sum(dim=0)
            value_bias_grads.append(grad)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (qkv_bias, proj_weight, proj_bias), grad in zip(
            self.value_bias_specs, value_bias_grads
        ):
=======
        for (
            qkv_bias,
            proj_weight,
            proj_bias,
            retained_dims,
        ), grad in zip(self.value_bias_specs, value_bias_grads):
>>>>>>> REPLACE

<<<<<<< SEARCH
            omitted_value = -step_size * exp_avg / denom
            proj_bias.add_(
                proj_weight[:, -omitted_value.numel():] @ omitted_value
            )
=======
            omitted_value = -step_size * exp_avg / denom
            omitted_dims = [
                dim
                for dim in range(proj_weight.size(1))
                if dim not in retained_dims
            ]
            proj_bias.add_(
                proj_weight[:, omitted_dims] @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    for qkv_bias, proj_weight, proj_bias in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_dims = 2 * proj_weight.size(1) - qkv_bias.numel()
        omitted_grad = (
            proj_weight.detach()[:, -omitted_dims:]
            * proj_bias.grad.detach().unsqueeze(1)
        ).sum(dim=0)
=======
    for qkv_bias, proj_weight, proj_bias, retained_dims in value_bias_specs:
        if proj_bias.grad is None:
            continue
        omitted_dims = [
            dim
            for dim in range(proj_weight.size(1))
            if dim not in retained_dims
        ]
        omitted_grad = (
            proj_weight.detach()[:, omitted_dims]
            * proj_bias.grad.detach().unsqueeze(1)
        ).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
        )
=======
        (
            block.attn.qkv.bias,
            block.attn.proj.weight,
            block.attn.proj.bias,
            (1,),
        )
>>>>>>> REPLACE