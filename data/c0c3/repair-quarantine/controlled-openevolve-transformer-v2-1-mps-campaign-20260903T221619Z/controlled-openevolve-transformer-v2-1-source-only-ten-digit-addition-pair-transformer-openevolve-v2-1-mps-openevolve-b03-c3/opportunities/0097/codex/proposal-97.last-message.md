MECHANISM: Orthogonal query-key head-space gauge

HYPOTHESIS: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while fixing one key-projection coordinate through a shared orthogonal query-key rotation preserves attention logits exactly and avoids the unstable scale and output-shift mechanisms tested previously.

INTENDED_EDIT: Replace the shared key projection with a 31-parameter Givens-gauge projection that fixes one coordinate to zero, rotates both query heads consistently at initialization, includes the omitted ambient gradient in clipping, and preserves the existing absorbed-scale optimizer path.

EVIDENCE: The current three-scale-absorption design achieved 99.39% at 1,378 parameters, whereas extending scale absorption or attention-output gauges failed; this tests a distinct exact symmetry of the learned dot-product attention while preserving its initialized function.

<<<<<<< SEARCH
        return F.linear(x, weight, full_bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, weight, full_bias)


class GaugeFixedKeyProjection(nn.Module):
    """Shared key projection with one orthogonal-basis gauge removed."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        q_proj: nn.Linear,
        q_bias: nn.Parameter,
        n_head: int,
    ):
        super().__init__()
        if out_features < 2:
            raise ValueError("key head dimension must be at least two")
        self.in_features = in_features
        self.out_features = out_features
        self.n_head = n_head
        object.__setattr__(self, "q_proj", q_proj)
        object.__setattr__(self, "q_bias", q_bias)
        self.first_column = nn.Parameter(torch.empty(out_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std=None) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        if std is None:
            nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        else:
            nn.init.normal_(raw_weight, mean=0.0, std=std)

        a = raw_weight[0, 0]
        b = raw_weight[1, 0]
        radius = torch.sqrt(a.square() + b.square()).clamp_min(
            torch.finfo(raw_weight.dtype).tiny
        )
        cosine = a / radius
        sine = b / radius

        rotated = raw_weight.clone()
        rotated[0].copy_(
            cosine * raw_weight[0] + sine * raw_weight[1]
        )
        rotated[1].copy_(
            -sine * raw_weight[0] + cosine * raw_weight[1]
        )
        self.first_column.copy_(
            torch.cat((rotated[:1, 0], rotated[2:, 0]))
        )
        self.weight_rest.copy_(rotated[:, 1:])

        q_weight = self.q_proj.weight.view(
            self.n_head, self.out_features, self.in_features
        )
        q_first = q_weight[:, 0, :].clone()
        q_second = q_weight[:, 1, :].clone()
        q_weight[:, 0, :].copy_(
            cosine * q_first + sine * q_second
        )
        q_weight[:, 1, :].copy_(
            -sine * q_first + cosine * q_second
        )

        q_bias = self.q_bias.view(self.n_head, self.out_features)
        bias_first = q_bias[:, 0].clone()
        bias_second = q_bias[:, 1].clone()
        q_bias[:, 0].copy_(
            cosine * bias_first + sine * bias_second
        )
        q_bias[:, 1].copy_(
            -sine * bias_first + cosine * bias_second
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_column = torch.cat(
            (
                self.first_column[:1],
                self.first_column.new_zeros(1),
                self.first_column[1:],
            )
        )
        full_weight = torch.cat(
            (first_column.unsqueeze(1), self.weight_rest),
            dim=1,
        )
        if torch.is_grad_enabled():
            full_weight.retain_grad()
            self.full_weight = full_weight
        return F.linear(x, full_weight)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve independent learned queries while broadcasting one learned
        # key/value representation across both attention maps.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        # Preserve independent learned queries while broadcasting one learned
        # key/value representation across both attention maps.
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.k_proj = GaugeFixedKeyProjection(
            d_model,
            self.head_dim,
            self.q_proj,
            self.q_bias,
            n_head,
        )
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
=======
        elif isinstance(module, GaugeFixedKeyProjection):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
=======
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight_rest[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
>>>>>>> REPLACE

<<<<<<< SEARCH
            model.token_emb.full_weight.grad.detach().reshape(-1),
        ]
        for blk in model.blocks:
            full_gauge_grads.extend(
                full_grad.detach()
                for full_grad in blk.attn.full_relative_bias.grad
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        attention_absorbed_grads = []
=======
            model.token_emb.full_weight.grad.detach().reshape(-1),
        ]
        for blk in model.blocks:
            full_gauge_grads.extend(
                full_grad.detach()
                for full_grad in blk.attn.full_relative_bias.grad
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )

        key_omitted_grads = [
            blk.attn.k_proj.full_weight.grad[1, 0].detach()
            for blk in model.blocks
        ]

        attention_absorbed_grads = []
>>>>>>> REPLACE

<<<<<<< SEARCH
                    blk.attn.q_proj.weight.grad[:, -3:],
                    blk.attn.k_proj.weight.grad[:, -3:],
                    blk.attn.v_proj.weight.grad[:, -3:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
=======
                    blk.attn.q_proj.weight.grad[:, -3:],
                    blk.attn.k_proj.weight_rest.grad[:, -3:],
                    blk.attn.v_proj.weight.grad[:, -3:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight_rest[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
>>>>>>> REPLACE

<<<<<<< SEARCH
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for (
=======
            for full_grad in full_gauge_grads:
                grad_sq = grad_sq + full_grad[-1].float().square()
            for omitted_grad in key_omitted_grads:
                grad_sq = grad_sq + omitted_grad.float().square()
            for (
>>>>>>> REPLACE

<<<<<<< SEARCH
            blk.attn.q_proj.weight.grad[:, -3:].zero_()
            blk.attn.k_proj.weight.grad[:, -3:].zero_()
            blk.attn.v_proj.weight.grad[:, -3:].zero_()
=======
            blk.attn.q_proj.weight.grad[:, -3:].zero_()
            blk.attn.k_proj.weight_rest.grad[:, -3:].zero_()
            blk.attn.v_proj.weight.grad[:, -3:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.attn.k_proj.weight[:, -3:].copy_(
                    effective_weight[q_end:k_end]
                )
=======
                blk.attn.k_proj.weight_rest[:, -3:].copy_(
                    effective_weight[q_end:k_end]
                )
>>>>>>> REPLACE