MECHANISM: Optimizer-preserving Q/K bilinear score folding

HYPOTHESIS: Folding query weights, key weights, and query bias into learned bilinear attention scores will reduce the model from 1,283 to 1,275 parameters while retaining at least 99% accuracy, because it preserves the exact attention function and reconstructs the successful full-width AdamW updates through the composite’s chain rule.

INTENDED_EDIT: Replace explicit query/key projections with per-head bilinear and linear score parameters, while maintaining virtual query, key, and query-bias states solely inside the quotient-aware optimizer.

EVIDENCE: The 1,397-parameter attention constant-folding model retained 99.93% after joint chain-rule reconstruction, and the current 1,283-parameter model retained 99.93% with optimizer-preserving quotients; this applies the same demonstrated technique to the exact Q/K factorization redundancy.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Q/K/V projections on LayerNorm's seven zero-mean coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.query_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.key_weight = nn.Parameter(torch.empty(d_model, d_model - 1))
        self.value_weight = nn.Parameter(torch.empty(d_model, d_model - 1))

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
        )
        return q, k, v
=======
class GaugeFixedQKV(nn.Module):
    """Folded Q/K scores and a value projection on zero-mean coordinates."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        reduced_width = d_model - 1
        self.score_weight = nn.Parameter(
            torch.empty(n_head, reduced_width, reduced_width)
        )
        self.score_bias = nn.Parameter(
            torch.empty(n_head, reduced_width)
        )
        self.value_weight = nn.Parameter(torch.empty(d_model, reduced_width))

        # Preserve the RNG stream of the removed bias-free QKV Linear.
        discarded_weight = torch.empty(3 * d_model, d_model)
        nn.init.kaiming_uniform_(discarded_weight, a=math.sqrt(5))

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
            self.score_weight.copy_(
                torch.einsum("hdf,hdg->hfg", query_heads, key_heads)
            )
            self.score_bias.copy_(
                torch.einsum("hd,hdg->hg", bias_heads, key_heads)
            )
            self.value_weight.copy_(
                value_weight[:, :-1] - value_weight[:, -1:]
            )

        # Consumed by QuotientAdamW to preserve the virtual factor updates.
        self._initial_query_weight = query_weight.detach().clone()
        self._initial_key_weight = key_weight.detach().clone()
        self._initial_query_bias = query_bias.detach().clone()
        self._initial_value_weight = value_weight.detach().clone()

    def forward(
        self, normalized_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        reduced_x = normalized_x[..., :-1]
        left_score = torch.einsum(
            "btf,hfg->bhtg", reduced_x, self.score_weight
        )
        att = torch.einsum(
            "bhtg,bsg->bhts", left_score, reduced_x
        )
        key_bias = torch.einsum(
            "hg,bsg->bhs", self.score_bias, reduced_x
        )
        att = att + key_bias.unsqueeze(2)
        v = F.linear(reduced_x, self.value_weight)
        return att, v
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = GaugeFixedQKV(d_model)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = GaugeFixedResidualProjection(d_model, d_model)
=======
        self.qkv = GaugeFixedQKV(d_model, n_head)
        self.proj = GaugeFixedResidualProjection(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = self.qkv(x, normalized_x)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        att, v = self.qkv(normalized_x)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = att / math.sqrt(self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
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
=======
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
            (block.mlp.fc2.weight, 0) for block in model.blocks
        ] + [
            (block.attn.rel_bias, 1) for block in model.blocks
        ]
        self.score_specs = [
            (
                block.attn.qkv.score_weight,
                block.attn.qkv.score_bias,
                block.attn.qkv,
            )
            for block in model.blocks
        ]
        self.attention_specs = [
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        self.gauge_params = [param for param, _ in self.gauge_specs]
        self.score_params = [
            param
            for score_weight, score_bias, _ in self.score_specs
            for param in (score_weight, score_bias)
        ]
        self.attention_params = [
            param
            for weight, _, proj_weight, folded_bias
            in self.attention_specs
            for param in (weight, proj_weight, folded_bias)
        ]
        excluded_ids = {
            id(param)
            for param in (
                self.gauge_params
                + self.score_params
                + self.attention_params
            )
        }
        self.custom_param_ids = {
            id(param)
            for param in self.score_params + self.attention_params
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.attention_states = []
        for weight, qkv, proj_weight, _ in self.attention_specs:
=======
        self.score_states = []
        for score_weight, _, qkv in self.score_specs:
            query_weight = qkv._initial_query_weight.to(
                device=score_weight.device, dtype=score_weight.dtype
            )
            key_weight = qkv._initial_key_weight.to(
                device=score_weight.device, dtype=score_weight.dtype
            )
            query_bias = qkv._initial_query_bias.to(
                device=score_weight.device, dtype=score_weight.dtype
            )
            delattr(qkv, "_initial_query_weight")
            delattr(qkv, "_initial_key_weight")
            delattr(qkv, "_initial_query_bias")

            full_shape = list(query_weight.shape)
            full_shape[1] += 1
            self.score_states.append(
                {
                    "step": 0,
                    "query_weight": query_weight,
                    "key_weight": key_weight,
                    "query_bias": query_bias,
                    "exp_avg_query": query_weight.new_zeros(full_shape),
                    "exp_avg_sq_query": query_weight.new_zeros(full_shape),
                    "exp_avg_key": key_weight.new_zeros(full_shape),
                    "exp_avg_sq_key": key_weight.new_zeros(full_shape),
                    "exp_avg_bias": query_bias.new_zeros(query_bias.shape),
                    "exp_avg_sq_bias": query_bias.new_zeros(
                        query_bias.shape
                    ),
                }
            )

        self.attention_states = []
        for weight, qkv, proj_weight, _ in self.attention_specs:
>>>>>>> REPLACE

<<<<<<< SEARCH
        for param in self.gauge_params + self.attention_params:
=======
        for param in (
            self.gauge_params + self.score_params + self.attention_params
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _full_attention_grads(
=======
    @staticmethod
    def _full_score_grads(
        score_weight_param, score_bias_param, state
    ):
        score_weight_grad = (
            torch.zeros_like(score_weight_param)
            if score_weight_param.grad is None
            else score_weight_param.grad.detach()
        )
        score_bias_grad = (
            torch.zeros_like(score_bias_param)
            if score_bias_param.grad is None
            else score_bias_param.grad.detach()
        )

        n_head, reduced_width, _ = score_weight_param.shape
        query_heads = state["query_weight"].view(
            n_head, -1, reduced_width
        )
        key_heads = state["key_weight"].view(
            n_head, -1, reduced_width
        )
        bias_heads = state["query_bias"].view(n_head, -1)

        query_grad = torch.einsum(
            "hfg,hdg->hdf", score_weight_grad, key_heads
        )
        key_grad = torch.einsum(
            "hdf,hfg->hdg", query_heads, score_weight_grad
        )
        key_grad = key_grad + (
            bias_heads.unsqueeze(-1) * score_bias_grad.unsqueeze(1)
        )
        bias_grad = torch.einsum(
            "hdg,hg->hd", key_heads, score_bias_grad
        ).reshape_as(state["query_bias"])

        query_grad = query_grad.reshape_as(state["query_weight"])
        key_grad = key_grad.reshape_as(state["key_weight"])
        full_query_grad = torch.cat(
            [query_grad, -query_grad.sum(dim=1, keepdim=True)], dim=1
        )
        full_key_grad = torch.cat(
            [key_grad, -key_grad.sum(dim=1, keepdim=True)], dim=1
        )
        return full_query_grad, full_key_grad, bias_grad

    @staticmethod
    def _full_attention_grads(
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Replace folded attention gradients with gradients of the virtual
        # value, LayerNorm, full projection, and shared-bias parameters.
=======
        # Replace composite score gradients with those of the virtual
        # full-width query, key, and query-bias parameters.
        for (
            score_weight,
            score_bias,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
                score_weight, score_bias, state
            )
            for grad in full_grads:
                total_sq.add_(grad.float().square().sum())

        # Replace folded attention gradients with gradients of the virtual
        # value, LayerNorm, full projection, and shared-bias parameters.
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        def update_quotient(
            value, full_grad, exp_avg, exp_avg_sq,
            bias_correction1, bias_correction2, axis
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
            stored_update = update.narrow(
                axis, 0, value.shape[axis]
            )
            reference_update = update.narrow(
                axis, value.shape[axis], 1
            )
            value.add_(
                stored_update - reference_update,
                alpha=-lr / bias_correction1,
            )

        for (
            score_weight,
            score_bias,
            _,
        ), state in zip(self.score_specs, self.score_states):
            if (
                score_weight.grad is None
                and score_bias.grad is None
            ):
                continue

            query_grad, key_grad, bias_grad = self._full_score_grads(
                score_weight, score_bias, state
            )
            state["step"] += 1
            step = state["step"]
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            update_quotient(
                state["query_weight"],
                query_grad,
                state["exp_avg_query"],
                state["exp_avg_sq_query"],
                bias_correction1,
                bias_correction2,
                1,
            )
            update_quotient(
                state["key_weight"],
                key_grad,
                state["exp_avg_key"],
                state["exp_avg_sq_key"],
                bias_correction1,
                bias_correction2,
                1,
            )
            update_virtual(
                state["query_bias"],
                bias_grad,
                state["exp_avg_bias"],
                state["exp_avg_sq_bias"],
                bias_correction1,
                bias_correction2,
            )

            n_head, reduced_width, _ = score_weight.shape
            query_heads = state["query_weight"].view(
                n_head, -1, reduced_width
            )
            key_heads = state["key_weight"].view(
                n_head, -1, reduced_width
            )
            bias_heads = state["query_bias"].view(n_head, -1)
            score_weight.copy_(
                torch.einsum("hdf,hdg->hfg", query_heads, key_heads)
            )
            score_bias.copy_(
                torch.einsum("hd,hdg->hg", bias_heads, key_heads)
            )

        for (
            weight,
            _,
            proj_weight,
            folded_bias,
        ), state in zip(
>>>>>>> REPLACE

<<<<<<< SEARCH
            update_quotient(
                proj_weight,
                proj_weight_grad,
                state["exp_avg_proj"],
                state["exp_avg_sq_proj"],
                bias_correction1,
                bias_correction2,
            )
=======
            update_quotient(
                proj_weight,
                proj_weight_grad,
                state["exp_avg_proj"],
                state["exp_avg_sq_proj"],
                bias_correction1,
                bias_correction2,
                0,
            )
>>>>>>> REPLACE