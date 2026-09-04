MECHANISM: Optimizer-preserving Q/K basis gauge quotient

HYPOTHESIS: Canonicalizing each head’s rank-four key projection to an identity basis will reduce the model from 1,275 to 1,251 parameters while retaining at least 99% accuracy, because it preserves the exact attention scores and reconstructs the successful virtual Q/K AdamW gradients through the gauge transformation.

INTENDED_EDIT: Replace the 112 stored bilinear-score parameters with 88 canonical query, key-tail, and query-bias parameters, while retaining full virtual query/key optimizer states and recanonicalizing after every update.

EVIDENCE: The 1,275-parameter folded Q/K model achieved 99.99% using virtual factor updates; the proposed 24-parameter reduction removes only the exact per-head 4-by-4 change-of-basis redundancy of those same factors.

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        reduced_x = normalized_x[..., :-1]
        key_prefix = torch.eye(
            self.head_dim,
            device=reduced_x.device,
            dtype=reduced_x.dtype,
        ).expand(self.n_head, -1, -1)
        key_weight = torch.cat([key_prefix, self.key_tail], dim=-1)
        q = torch.einsum(
            "btf,hdf->bhtd", reduced_x, self.query_weight
        )
        q = q + self.query_bias.unsqueeze(0).unsqueeze(2)
        k = torch.einsum(
            "bsf,hdf->bhsd", reduced_x, key_weight
        )
        att = torch.einsum("bhtd,bhsd->bhts", q, k)
        v = F.linear(reduced_x, self.value_weight)
        return att, v
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.score_specs = [
            (
                block.attn.qkv.score_weight,
                block.attn.qkv.score_bias,
                block.attn.qkv,
            )
            for block in model.blocks
        ]
=======
        self.score_specs = [
            (
                block.attn.qkv.query_weight,
                block.attn.qkv.key_tail,
                block.attn.qkv.query_bias,
                block.attn.qkv,
            )
            for block in model.blocks
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.score_params = [
            param
            for score_weight, score_bias, _ in self.score_specs
            for param in (score_weight, score_bias)
        ]
=======
        self.score_params = [
            param
            for query_weight, key_tail, query_bias, _
            in self.score_specs
            for param in (query_weight, key_tail, query_bias)
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        self.score_states = []
        for query_param, _, _, qkv in self.score_specs:
            query_weight = qkv._initial_query_weight.to(
                device=query_param.device, dtype=query_param.dtype
            )
            key_weight = qkv._initial_key_weight.to(
                device=query_param.device, dtype=query_param.dtype
            )
            query_bias = qkv._initial_query_bias.to(
                device=query_param.device, dtype=query_param.dtype
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    @staticmethod
    def _full_score_grads(
        query_param, key_tail_param, bias_param, state
    ):
        canonical_query_grad = (
            torch.zeros_like(query_param)
            if query_param.grad is None
            else query_param.grad.detach()
        )
        canonical_key_tail_grad = (
            torch.zeros_like(key_tail_param)
            if key_tail_param.grad is None
            else key_tail_param.grad.detach()
        )
        canonical_bias_grad = (
            torch.zeros_like(bias_param)
            if bias_param.grad is None
            else bias_param.grad.detach()
        )

        n_head, head_dim, reduced_width = query_param.shape
        virtual_key_heads = state["key_weight"].view(
            n_head, head_dim, reduced_width
        )
        key_basis = virtual_key_heads[..., :head_dim]

        # Recover the gradient of the fixed identity key block from the
        # exact GL(head_dim) factorization invariance.
        canonical_key_prefix_grad = torch.matmul(
            query_param, canonical_query_grad.transpose(-1, -2)
        )
        canonical_key_prefix_grad = canonical_key_prefix_grad + (
            bias_param.unsqueeze(-1)
            * canonical_bias_grad.unsqueeze(-2)
        )
        canonical_key_prefix_grad = canonical_key_prefix_grad - (
            torch.matmul(
                canonical_key_tail_grad,
                key_tail_param.transpose(-1, -2),
            )
        )
        canonical_key_grad = torch.cat(
            [canonical_key_prefix_grad, canonical_key_tail_grad],
            dim=-1,
        )

        # Map canonical-factor gradients back to the virtual Q/K factors
        # whose full-width AdamW moments are retained.
        query_grad = torch.matmul(
            key_basis, canonical_query_grad
        )
        key_grad = torch.linalg.solve(
            key_basis.transpose(-1, -2), canonical_key_grad
        )
        bias_grad = torch.matmul(
            key_basis, canonical_bias_grad.unsqueeze(-1)
        ).squeeze(-1)

        query_grad = query_grad.reshape_as(state["query_weight"])
        key_grad = key_grad.reshape_as(state["key_weight"])
        bias_grad = bias_grad.reshape_as(state["query_bias"])
        full_query_grad = torch.cat(
            [query_grad, -query_grad.sum(dim=1, keepdim=True)], dim=1
        )
        full_key_grad = torch.cat(
            [key_grad, -key_grad.sum(dim=1, keepdim=True)], dim=1
        )
        return full_query_grad, full_key_grad, bias_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
        for (
            score_weight,
            score_bias,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
                score_weight, score_bias, state
            )
=======
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            full_grads = self._full_score_grads(
                query_param, key_tail_param, bias_param, state
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        for (
            query_param,
            key_tail_param,
            bias_param,
            _,
        ), state in zip(self.score_specs, self.score_states):
            if (
                query_param.grad is None
                and key_tail_param.grad is None
                and bias_param.grad is None
            ):
                continue

            query_grad, key_grad, bias_grad = self._full_score_grads(
                query_param, key_tail_param, bias_param, state
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

            n_head, head_dim, reduced_width = query_param.shape
            query_heads = state["query_weight"].view(
                n_head, head_dim, reduced_width
            )
            key_heads = state["key_weight"].view(
                n_head, head_dim, reduced_width
            )
            bias_heads = state["query_bias"].view(
                n_head, head_dim
            )
            key_basis = key_heads[..., :head_dim]
            query_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2), query_heads
                )
            )
            key_tail_param.copy_(
                torch.linalg.solve(
                    key_basis, key_heads[..., head_dim:]
                )
            )
            bias_param.copy_(
                torch.matmul(
                    key_basis.transpose(-1, -2),
                    bias_heads.unsqueeze(-1),
                ).squeeze(-1)
            )
>>>>>>> REPLACE