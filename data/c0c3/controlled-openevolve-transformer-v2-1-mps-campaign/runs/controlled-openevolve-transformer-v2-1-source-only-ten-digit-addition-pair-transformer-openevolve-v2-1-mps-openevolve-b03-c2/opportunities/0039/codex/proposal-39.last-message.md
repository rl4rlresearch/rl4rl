MECHANISM: LayerNorm-nullspace input-weight quotient

HYPOTHESIS: Applying the qualified residual-output quotients and removing the feature-uniform input coordinate from every QKV and `fc1` row will produce a 1514-parameter model with at least 99% accuracy, because non-affine LayerNorm outputs have zero feature mean and the optimizer reconstructs the omitted coordinates for folded-factor AdamW and clipping.

INTENDED_EDIT: Store attention and MLP residual-output weights with seven relative output coordinates, and store all 24 QKV plus 12 `fc1` rows with seven relative input coordinates while maintaining their full-coordinate optimizer dynamics.

EVIDENCE: Reference Design 3 achieved 100% accuracy at 1550 parameters after quotienting all attention and MLP residual-output columns. Its QKV and `fc1` projections both consume non-affine LayerNorm outputs, making one feature-uniform input direction per row functionally null and motivating the additional 36-parameter reduction.

<<<<<<< SEARCH
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
        return y
=======
        qkv_weight_relative = torch.cat(
            (
                self.qkv.weight,
                self.qkv.weight.new_zeros(
                    (self.qkv.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        qkv_weight = (
            qkv_weight_relative
            + self.qkv.weight.mean(dim=-1, keepdim=True)
        )
        qkv = F.linear(x, qkv_weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        weight_relative = torch.cat(
            (
                self.proj.weight,
                self.proj.weight.new_zeros(
                    (self.proj.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        proj_weight = (
            weight_relative
            + self.proj.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        proj_bias = relative_bias + self.proj.bias.mean()
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
        return y
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, self.fc2.weight, fc2_bias))
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fc1_weight_relative = torch.cat(
            (
                self.fc1.weight,
                self.fc1.weight.new_zeros(
                    (self.fc1.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc1_weight = (
            fc1_weight_relative
            + self.fc1.weight.mean(dim=-1, keepdim=True)
        )
        hidden = F.gelu(F.linear(x, fc1_weight, self.fc1.bias))
        weight_relative = torch.cat(
            (
                self.fc2.weight,
                self.fc2.weight.new_zeros(
                    (self.fc2.weight.size(0), 1)
                ),
            ),
            dim=-1,
        )
        fc2_weight = (
            weight_relative
            + self.fc2.weight.mean(dim=-1, keepdim=True)
        ).transpose(0, 1)
        relative_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        fc2_bias = relative_bias + self.fc2.bias.mean()
        return self.drop(F.linear(hidden, fc2_weight, fc2_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # A global feature-uniform shift of every tied token row is canceled
=======
        self.apply(self._init_weights)

        # LayerNorm-null input coordinates and final-LayerNorm-null output
        # coordinates are retained only through relative representatives.
        for block in self.blocks:
            full_qkv_weight = block.attn.qkv.weight.detach()
            block.attn.qkv.weight = nn.Parameter(
                full_qkv_weight[:, :-1] - full_qkv_weight[:, -1:]
            )
            full_proj_weight = (
                block.attn.proj.weight.detach().transpose(0, 1)
            )
            block.attn.proj.weight = nn.Parameter(
                full_proj_weight[:, :-1] - full_proj_weight[:, -1:]
            )
            full_fc1_weight = block.mlp.fc1.weight.detach()
            block.mlp.fc1.weight = nn.Parameter(
                full_fc1_weight[:, :-1] - full_fc1_weight[:, -1:]
            )
            full_fc2_weight = (
                block.mlp.fc2.weight.detach().transpose(0, 1)
            )
            block.mlp.fc2.weight = nn.Parameter(
                full_fc2_weight[:, :-1] - full_fc2_weight[:, -1:]
            )

        # A global feature-uniform shift of every tied token row is canceled
>>>>>>> REPLACE

<<<<<<< SEARCH
    return min_lr + (base_lr - min_lr) * cosine


class QuotientAdamW(torch.optim.AdamW):
=======
    return min_lr + (base_lr - min_lr) * cosine


def reconstruct_input_weight(param):
    relative = torch.cat(
        (
            param,
            param.new_zeros((param.size(0), 1)),
        ),
        dim=-1,
    )
    return relative + param.mean(dim=-1, keepdim=True)


def reconstruct_output_weight(param):
    return reconstruct_input_weight(param).transpose(0, 1)


class QuotientAdamW(torch.optim.AdamW):
>>>>>>> REPLACE

<<<<<<< SEARCH
        if "factor_weight" not in state:
            state["factor_step"] = 0
            state["factor_weight"] = param.detach().clone()
            state["factor_scale"] = param.new_ones(param.size(1))
            state["factor_weight_exp_avg"] = torch.zeros_like(param)
            state["factor_weight_exp_avg_sq"] = torch.zeros_like(param)
            state["factor_scale_exp_avg"] = param.new_zeros(
                param.size(1)
            )
            state["factor_scale_exp_avg_sq"] = param.new_zeros(
                param.size(1)
            )
        return state

    @torch.no_grad()
    def factor_grad_sq(self, param):
        state = self._factor_state(param)
        grad = param.grad.detach()
        weight_grad = grad * state["factor_scale"].unsqueeze(0)
        scale_grad = (
            grad * state["factor_weight"]
        ).sum(dim=0)
        return weight_grad.square().sum() + scale_grad.square().sum()
=======
        if "factor_weight" not in state:
            state["factor_step"] = 0
            state["factor_weight"] = reconstruct_input_weight(
                param.detach()
            ).clone()
            state["factor_scale"] = param.new_ones(param.size(1) + 1)
            state["factor_weight_exp_avg"] = torch.zeros_like(
                state["factor_weight"]
            )
            state["factor_weight_exp_avg_sq"] = torch.zeros_like(
                state["factor_weight"]
            )
            state["factor_scale_exp_avg"] = param.new_zeros(
                param.size(1) + 1
            )
            state["factor_scale_exp_avg_sq"] = param.new_zeros(
                param.size(1) + 1
            )
        return state

    @torch.no_grad()
    def factor_grad_sq(self, param):
        state = self._factor_state(param)
        grad = param.grad.detach()
        full_grad = torch.cat(
            (grad, -grad.sum(dim=-1, keepdim=True)),
            dim=-1,
        )
        weight_grad = (
            full_grad * state["factor_scale"].unsqueeze(0)
        )
        scale_grad = (
            full_grad * state["factor_weight"]
        ).sum(dim=0)
        return weight_grad.square().sum() + scale_grad.square().sum()
>>>>>>> REPLACE

<<<<<<< SEARCH
                grad = (
                    proj_weight.detach()
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
=======
                full_proj_weight = reconstruct_output_weight(
                    proj_weight.detach()
                )
                grad = (
                    full_proj_weight
                    * full_proj_grad.unsqueeze(1)
                ).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
            factor_weight = state["factor_weight"]
            factor_scale = state["factor_scale"]
            weight_grad = grad * factor_scale.unsqueeze(0)
            scale_grad = (grad * factor_weight).sum(dim=0)
=======
            factor_weight = state["factor_weight"]
            factor_scale = state["factor_scale"]
            full_grad = torch.cat(
                (grad, -grad.sum(dim=-1, keepdim=True)),
                dim=-1,
            )
            weight_grad = full_grad * factor_scale.unsqueeze(0)
            scale_grad = (full_grad * factor_weight).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
            param.copy_(
                factor_weight * factor_scale.unsqueeze(0)
            )
=======
            full_product = (
                factor_weight * factor_scale.unsqueeze(0)
            )
            param.copy_(
                full_product[:, :-1] - full_product[:, -1:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_delta = proj_weight @ omitted_value
=======
            full_delta = (
                reconstruct_output_weight(proj_weight)
                @ omitted_value
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        omitted_grad = (
            proj_weight.detach()
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
=======
        full_proj_weight = reconstruct_output_weight(
            proj_weight.detach()
        )
        omitted_grad = (
            full_proj_weight
            * full_proj_grad.unsqueeze(1)
        ).sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
    ]
=======
    quotient_params = [model.token_emb.weight, model.pos_emb.weight] + [
        block.attn.proj.bias for block in model.blocks
    ] + [
        block.attn.proj.weight for block in model.blocks
    ] + [
        block.mlp.fc2.bias for block in model.blocks
    ] + [
        block.mlp.fc2.weight for block in model.blocks
    ]
>>>>>>> REPLACE