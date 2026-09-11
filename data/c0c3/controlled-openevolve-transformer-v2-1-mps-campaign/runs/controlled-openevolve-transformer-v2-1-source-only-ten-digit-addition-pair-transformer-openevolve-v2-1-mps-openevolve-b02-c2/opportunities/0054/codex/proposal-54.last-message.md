MECHANISM: Full-coordinate Adam on an orthonormal residual-bias quotient

HYPOTHESIS: Removing the final MLP bias’s uniform direction while updating its seven coordinates through reconstructed eight-coordinate AdamW dynamics will produce a 1,621-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified 1,622-parameter design, express the final `fc2` bias in a seven-dimensional zero-mean basis, and emulate the parent model’s full-coordinate AdamW update for that quotient parameter.

EVIDENCE: The 1,622-parameter parent reached 99.92%, while the same orthonormal residual-bias quotient with ordinary basis-coordinate AdamW reached 98.04%; this isolates optimizer-coordinate geometry as the most informative remaining issue.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """QKV projection retaining 2:3 query biases, no key biases, and 0:1 value biases."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        compact_bias = torch.cat(
            (
                linear.bias[: self.head_dim - 2],
                linear.bias[self.head_dim : key_start - 1],
                linear.bias[self.value_start + self.head_dim + 3 :],
            )
        )
        self.bias = nn.Parameter(compact_bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_query_end = self.head_dim - 2
        second_query_end = first_query_end + self.head_dim - 1
        full_bias = torch.cat(
            (
                self.bias[:first_query_end],
                self.bias.new_zeros(2),
                self.bias[first_query_end:second_query_end],
                self.bias.new_zeros(1),
                self.bias.new_zeros(self.key_start),
                self.bias.new_zeros(self.head_dim),
                self.bias.new_zeros(3),
                self.bias[second_query_end:],
            )
        )
        return F.linear(x, self.weight, full_bias)
=======
class CompactQKV(nn.Module):
    """QKV projection with shared head-0 queries and one retained value bias."""

    def __init__(self, linear: nn.Linear, key_start: int, second_head_start: int):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.weight = linear.weight
        query_bias = torch.cat(
            (
                linear.bias[:1],
                linear.bias[self.head_dim : key_start - 1],
            )
        )
        self.query_bias = nn.Parameter(query_bias.detach().clone())
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.query_bias[:1].expand(self.head_dim - 2),
                self.query_bias.new_zeros(2),
                self.query_bias[1:],
                self.query_bias.new_zeros(1),
                self.query_bias.new_zeros(self.key_start),
                self.query_bias.new_zeros(self.head_dim),
                self.query_bias.new_zeros(3),
                self.value_bias,
            )
        )
        return F.linear(x, self.weight, full_bias)


class CompactSharedProjection(nn.Module):
    """Attention projection sharing its final bias with the retained value bias."""

    def __init__(self, linear: nn.Linear, shared_bias: nn.Parameter):
        super().__init__()
        self.weight = linear.weight
        self.bias = nn.Parameter(linear.bias[:-1].detach().clone())
        self.shared_bias = shared_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.shared_bias))
        return F.linear(x, self.weight, full_bias)


class CompactLayerNormBias(nn.Module):
    """LayerNorm with one downstream-linear-absorbed bias coordinate fixed."""

    def __init__(self, layer_norm: nn.LayerNorm):
        super().__init__()
        self.normalized_shape = layer_norm.normalized_shape
        self.eps = layer_norm.eps
        self.weight = layer_norm.weight
        self.bias = nn.Parameter(layer_norm.bias[:-1].detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            full_bias,
            self.eps,
        )


class CompactResidualProjection(nn.Module):
    """Linear projection with its downstream-LayerNorm-null bias direction removed."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight

        centered = torch.eye(self.out_features)[:, :-1]
        centered = centered - centered.mean(dim=0, keepdim=True)
        basis = torch.linalg.qr(centered, mode="reduced").Q
        self.register_buffer("bias_basis", basis, persistent=False)
        self.bias = nn.Parameter(
            (basis.transpose(0, 1) @ linear.bias.detach()).clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = self.bias_basis @ self.bias
        return F.linear(x, self.weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = CompactLayerNormBias(nn.LayerNorm(cfg.d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve every baseline initialization draw before retaining two
        # head-0 and three head-1 query biases, no key biases, and one value bias.
        for block in self.blocks:
            block.attn.qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
=======
        # Retain the qualified QKV layout, then quotient its constant-offset
        # redundancy by sharing the final value and projection bias scalar.
        for block in self.blocks:
            compact_qkv = CompactQKV(
                block.attn.qkv,
                cfg.d_model,
                cfg.d_model + block.attn.head_dim,
            )
            block.attn.qkv = compact_qkv
            block.attn.proj = CompactSharedProjection(
                block.attn.proj,
                compact_qkv.value_bias,
            )

        # The final MLP residual is followed directly by the final LayerNorm,
        # so its uniform bias direction is functionally null.
        final_mlp = self.blocks[-1].mlp
        final_mlp.fc2 = CompactResidualProjection(final_mlp.fc2)
>>>>>>> REPLACE

<<<<<<< SEARCH
from src.model import ModelConfig, TinyDecoderLM, count_parameters
=======
from src.model import (
    CompactResidualProjection,
    ModelConfig,
    TinyDecoderLM,
    count_parameters,
)
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    quotient_modules = [
        module
        for module in model.modules()
        if isinstance(module, CompactResidualProjection)
    ]
    quotient_param_ids = {id(module.bias) for module in quotient_modules}
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in quotient_param_ids
    ]

    adam_betas = (0.9, 0.999)
    adam_eps = 1e-8
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        betas=adam_betas,
        eps=adam_eps,
    )
    quotient_states = [
        {
            "step": 0,
            "exp_avg": module.bias_basis.new_zeros(module.out_features),
            "exp_avg_sq": module.bias_basis.new_zeros(module.out_features),
        }
        for module in quotient_modules
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        for module in quotient_modules:
            module.bias.grad = None
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        # Reconstruct the quotient gradient in the original eight-coordinate
        # bias space so AdamW retains the qualified parent's coordinate dynamics.
        with torch.no_grad():
            beta1, beta2 = adam_betas
            for module, state in zip(quotient_modules, quotient_states):
                if module.bias.grad is None:
                    continue

                full_grad = module.bias_basis @ module.bias.grad
                state["step"] += 1
                state["exp_avg"].mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad,
                    full_grad,
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                full_denom = (
                    state["exp_avg_sq"].sqrt()
                    / math.sqrt(bias_correction2)
                ).add_(adam_eps)
                full_update = (
                    state["exp_avg"] / bias_correction1 / full_denom
                )

                module.bias.mul_(1.0 - lr_now * train_cfg.weight_decay)
                module.bias.add_(
                    module.bias_basis.transpose(0, 1) @ full_update,
                    alpha=-lr_now,
                )
>>>>>>> REPLACE