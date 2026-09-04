MECHANISM: Optimizer-decoupled null-direction reuse

HYPOTHESIS: Explicitly separating the second attention-output column into centered coefficients and a scalar mean will yield a 1,267-parameter model with at least 99% accuracy, because the reused final-LayerNorm scale will no longer be contaminated by AdamW updates to the column’s functional centered component.

INTENDED_EDIT: Reparameterize the second attention-output column with seven orthonormal mean-zero coefficients plus one explicit common scalar, reuse that scalar as final-LayerNorm scale coordinate 2, and retain the successful first-column mean reuse unchanged.

EVIDENCE: Implicitly reusing two dense column means produced only 15.95% accuracy, while one dense-column reuse achieved 100%. The explicit chart preserves the same full function class and fresh initialization but isolates the second null scalar in optimizer coordinates, directly testing whether dense-coordinate coupling caused the dual-reuse failure.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with one trainable scale stored in an attention null direction."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 6))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(
        self, x: torch.Tensor, shared_scale: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                self.weight[1:2],
                self.weight.new_ones(2),
                shared_scale.reshape(1),
                self.weight.new_ones(2),
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(self.bias, (0, 7)),
        )


class CausalSelfAttention(nn.Module):
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with two trainable scales stored in attention null directions."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 7))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(
        self, x: torch.Tensor, shared_scales: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:1],
                self.weight.new_ones(1),
                shared_scales[1:2],
                self.weight.new_ones(2),
                shared_scales[:1],
                self.weight.new_ones(2),
            )
        )
        return F.layer_norm(
            x,
            self.normalized_shape,
            weight,
            F.pad(self.bias, (0, 7)),
        )


class ExplicitSecondMeanLinear(nn.Module):
    """Square linear map with the second column mean as an explicit scalar."""

    def __init__(self, d_model: int):
        super().__init__()
        center_columns = []
        for j in range(d_model - 1):
            vector = torch.zeros(d_model)
            vector[: j + 1] = 1.0
            vector[j + 1] = -(j + 1)
            vector /= math.sqrt((j + 1) * (j + 2))
            center_columns.append(vector)
        self.register_buffer(
            "center_basis",
            torch.stack(center_columns, dim=1),
            persistent=False,
        )

        self.weight_without_second = nn.Parameter(
            torch.empty(d_model, d_model - 1)
        )
        self.second_center = nn.Parameter(torch.empty(d_model - 1))
        self.second_mean = nn.Parameter(torch.ones(()))
        self.bias = nn.Parameter(torch.empty(d_model))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            d_model = self.bias.numel()
            dense = torch.empty(
                d_model,
                d_model,
                device=self.bias.device,
                dtype=self.bias.dtype,
            )
            nn.init.normal_(dense, mean=0.0, std=0.02)
            self.weight_without_second[:, :1].copy_(dense[:, :1])
            self.weight_without_second[:, 1:].copy_(dense[:, 2:])
            self.second_center.copy_(
                self.center_basis.transpose(0, 1) @ dense[:, 1]
            )
            self.second_mean.fill_(1.0)
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        second_column = (
            self.center_basis @ self.second_center + self.second_mean
        )
        weight = torch.cat(
            (
                self.weight_without_second[:, :1],
                second_column.unsqueeze(1),
                self.weight_without_second[:, 1:],
            ),
            dim=1,
        )
        return F.linear(x, weight, self.bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
        self.proj = ExplicitSecondMeanLinear(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        with torch.no_grad():
            shared_column = self.blocks[-1].attn.proj.weight[:, 0]
            shared_column.add_(1.0 - shared_column.mean())
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            shared_column = (
                self.blocks[-1].attn.proj.weight_without_second[:, 0]
            )
            shared_column.add_(1.0 - shared_column.mean())
        nn.init.normal_(self.token_emb, mean=0.0, std=0.02)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, SingleRotationGaugeQKV):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        if isinstance(module, (SingleRotationGaugeQKV, ExplicitSecondMeanLinear)):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
        shared_final_scale = self.blocks[-1].attn.proj.weight[:, 0].mean()
        x = self.ln_f(x, shared_final_scale)
=======
        final_proj = self.blocks[-1].attn.proj
        shared_final_scales = torch.stack(
            (
                final_proj.weight_without_second[:, 0].mean(),
                final_proj.second_mean,
            )
        )
        x = self.ln_f(x, shared_final_scales)
>>>>>>> REPLACE