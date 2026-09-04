MECHANISM: Ambient-Adam row-gauge fixing of the folded MLP input weight

HYPOTHESIS: Removing one softmax-independent row-shift coordinate from each of the 12 `fc1` rows will reduce the qualified model from 1,478 to 1,466 learned parameters while retaining at least 99% accuracy, because bias-free LayerNorm produces mean-zero inputs and full eight-coordinate MLP-weight and LayerNorm-scale AdamW dynamics remain represented during training.

INTENDED_EDIT: Store each folded `fc1` row as seven differences with an anchored eighth coordinate, preserve its original full-width initialization and optimizer state in non-parameter tensors, and canonicalize the folded ambient product after every update.

EVIDENCE: The current 1,478-parameter design achieved 99.88% after exactly folding `ln2` scales into `fc1` with ambient-coordinate AdamW; the proposed reduction acts on that same qualified interface and removes an exact null direction induced by its bias-free LayerNorm, avoiding the capacity-reducing approximations that failed in prior trials.

<<<<<<< SEARCH
        return F.linear(x, self.weight, full_bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, self.weight, full_bias)


class GaugeFixedMeanZeroLinear(nn.Module):
    """Linear layer with row shifts removed for mean-zero inputs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            "ambient_init",
            torch.empty(out_features, in_features),
            persistent=False,
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.ambient_init, a=math.sqrt(5)
        )
        self.weight.copy_(
            self.ambient_init[:, :-1]
            - self.ambient_init[:, -1:]
        )
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
            self.ambient_init
        )
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def reset_normal_parameters(self, std: float) -> None:
        nn.init.normal_(
            self.ambient_init, mean=0.0, std=std
        )
        self.weight.copy_(
            self.ambient_init[:, :-1]
            - self.ambient_init[:, -1:]
        )
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight = torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.out_features, 1),
            ),
            dim=1,
        )
        if torch.is_grad_enabled():
            full_weight.retain_grad()
            self.full_weight = full_weight
        return F.linear(x, full_weight, self.bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
=======
        self.fc1 = GaugeFixedMeanZeroLinear(d_model, d_ff)
        self.fc2 = GaugeFixedBiasLinear(d_ff, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, GaugeFixedBiasLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedMeanZeroLinear):
            module.reset_normal_parameters(std=0.02)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
    value_attentions = [blk.attn for blk in model.blocks]
    mlp_weight_params = [blk.mlp.fc1.weight for blk in model.blocks]
    mlp_weight_ids = {id(p) for p in mlp_weight_params}
=======
    value_attentions = [blk.attn for blk in model.blocks]
    mlp_linears = [blk.mlp.fc1 for blk in model.blocks]
    mlp_weight_params = [linear.weight for linear in mlp_linears]
    mlp_weight_ids = {id(p) for p in mlp_weight_params}
>>>>>>> REPLACE

<<<<<<< SEARCH
    # These tensors are optimizer state for the eliminated ln2 scales and
    # their ambient fc1 weights. The model always stores their exact product.
    mlp_ambient_weights = [
        p.detach().clone() for p in mlp_weight_params
    ]
    mlp_ambient_scales = [
        torch.ones(p.shape[1], device=device, dtype=p.dtype)
        for p in mlp_weight_params
    ]
=======
    # These tensors retain the full fc1 rows and eliminated ln2 scales.
    # The learned matrix stores their product modulo row-shift gauges.
    mlp_ambient_weights = [
        linear.ambient_init.detach().clone()
        for linear in mlp_linears
    ]
    mlp_ambient_scales = [
        torch.ones(
            weight.shape[1],
            device=device,
            dtype=weight.dtype,
        )
        for weight in mlp_ambient_weights
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        effective_mlp_grads = [
            p.grad.detach().clone() for p in mlp_weight_params
        ]
=======
        effective_mlp_grads = [
            linear.full_weight.grad.detach().clone()
            for linear in mlp_linears
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
                stored_weight.copy_(
                    ambient_weight * ambient_scale.unsqueeze(0)
                )
=======
                effective_weight = (
                    ambient_weight * ambient_scale.unsqueeze(0)
                )
                stored_weight.copy_(
                    effective_weight[:, :-1]
                    - effective_weight[:, -1:]
                )
>>>>>>> REPLACE