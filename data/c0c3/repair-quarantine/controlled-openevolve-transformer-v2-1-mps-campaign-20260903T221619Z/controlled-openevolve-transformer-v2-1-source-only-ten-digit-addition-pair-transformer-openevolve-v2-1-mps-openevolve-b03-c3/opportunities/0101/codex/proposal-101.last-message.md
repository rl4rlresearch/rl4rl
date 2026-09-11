MECHANISM: Orthogonal value/output basis gauge

HYPOTHESIS: A 1,377-parameter model will retain at least 99% accuracy because it preserves the qualified 1,378-parameter model’s initial function and full four-dimensional value stream while removing one exact value-basis redundancy.

INTENDED_EDIT: Replace the shared value projection with a 31-parameter projection whose first value-column pair is orthogonally rotated to fix one coordinate at zero, compensate both attention-output head blocks at initialization, and adapt the existing absorbed-scale optimizer path to the reconstructed weight.

EVIDENCE: The current 1,378-parameter design reached 99.39%; prior 1,377-parameter failures modified LayerNorm, attention-output, MLP, or query-key gauges, while narrowing the value stream did not finish. This tests an untried exact value/output symmetry without reducing value width or content addressing.

<<<<<<< SEARCH
        return F.linear(x, weight, full_bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, weight, full_bias)

    @torch.no_grad()
    def rotate_value_basis_(self, rotation: torch.Tensor) -> None:
        """Compensate an orthogonal change of basis in every value head."""
        prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        weight = torch.cat(
            (
                torch.stack(prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        head_dim = rotation.size(0)
        rotated = torch.cat(
            [
                weight[:, start : start + head_dim]
                @ rotation.transpose(0, 1)
                for start in range(0, self.in_features, head_dim)
            ],
            dim=1,
        )
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                rotated[:-1, column] - rotated[-1, column]
            )
        self.weight_rest.copy_(
            rotated[:, len(self.weight_prefix) :]
        )


class GaugeFixedValueProjection(nn.Module):
    """Value projection with one orthogonal basis gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")
        self.in_features = in_features
        self.out_features = out_features
        self.first_column = nn.Parameter(
            torch.empty(out_features - 1)
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.full_weight = None
        self.initial_rotation = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = None) -> None:
        raw = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        if std is None:
            nn.init.kaiming_uniform_(raw, a=math.sqrt(5))
        else:
            nn.init.normal_(raw, mean=0.0, std=std)

        pair = raw[:2, 0]
        radius = pair.square().sum().sqrt()
        rotation = torch.eye(
            self.out_features, device=raw.device, dtype=raw.dtype
        )
        if float(radius.item()) > 0.0:
            cosine = pair[0] / radius
            sine = pair[1] / radius
            rotation[0, 0] = cosine
            rotation[0, 1] = sine
            rotation[1, 0] = -sine
            rotation[1, 1] = cosine

        rotated = rotation @ raw
        self.first_column.copy_(
            torch.cat((rotated[:1, 0], rotated[2:, 0]))
        )
        self.weight_rest.copy_(rotated[:, 1:])
        self.initial_rotation = rotation

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
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.v_proj = GaugeFixedValueProjection(
            d_model, self.head_dim
        )
        self.q_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)
        for blk in self.blocks:
            blk.attn.proj.rotate_value_basis_(
                blk.attn.v_proj.initial_rotation
            )

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
=======
        elif isinstance(module, GaugeFixedValueProjection):
            module.reset_parameters(std=0.02)
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
=======
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight_rest[:, -3:],
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            effective_grad = torch.cat(
                (
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
                ),
                dim=0,
            ).detach()
=======
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -3:],
                    blk.attn.k_proj.weight.grad[:, -3:],
                    blk.attn.v_proj.full_weight.grad[:, -3:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.full_weight[:, -3:],
                ),
                dim=0,
            ).detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -3:].zero_()
            blk.attn.k_proj.weight.grad[:, -3:].zero_()
            blk.attn.v_proj.weight.grad[:, -3:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
=======
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -3:].zero_()
            blk.attn.k_proj.weight.grad[:, -3:].zero_()
            blk.attn.v_proj.weight_rest.grad[:, -3:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.attn.v_proj.weight[:, -3:].copy_(
                    effective_weight[k_end:]
                )
=======
                blk.attn.v_proj.weight_rest[:, -3:].copy_(
                    effective_weight[k_end:]
                )
>>>>>>> REPLACE