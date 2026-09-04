MECHANISM: Ambient row-space quotienting after non-affine LayerNorm

HYPOTHESIS: Reparameterizing every `qkv` and `fc1` weight row with seven learned coordinates will reduce the model from 1430 to 1394 parameters while retaining at least 99% accuracy, because non-affine `ln1` and `ln2` produce zero-mean inputs, making all 36 common-mode row coordinates unobservable, while eight-dimensional ambient AdamW preserves the established optimization geometry.

INTENDED_EDIT: Replace the dense `qkv` and `fc1` weights with initialization-preserving zero-mean row parameterizations and extend the ambient optimizer to support their batched gauge coordinates.

EVIDENCE: Removing `ln2` affine scales retained 99.93% accuracy and removing `ln1` scales retained 99.49%; those changes expose exact rowwise input-shift symmetries, while ambient AdamW has already preserved accuracy across every tested initialization-preserving projection quotient.

<<<<<<< SEARCH
class OneColumnGaugeLinear(nn.Module):
=======
class LayerNormInputLinear(nn.Module):
    """Linear map quotiented by the common-mode direction of LayerNorm inputs."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coordinates = nn.Parameter(torch.empty(out_features, in_features - 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        inv_sqrt = in_features ** -0.5
        reflector = torch.full((in_features,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Match Linear's constructor-time draws so downstream initialization
        # retains the verified random-number stream.
        conceptual_weight = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(conceptual_weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)
        self._set_weight(conceptual_weight)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = 2.0 * (value @ self.reflector) / self.reflector_norm_sq
        return value - projection.unsqueeze(-1) * self.reflector

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(conceptual_weight)
            self.coordinates.copy_(transformed[:, 1:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.coordinates.new_empty(
            self.out_features, self.in_features
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def materialized_weight(self) -> torch.Tensor:
        transformed = F.pad(self.coordinates, (1, 0))
        return self._householder(transformed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.materialized_weight(), self.bias)


class OneColumnGaugeLinear(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
=======
        self.qkv = LayerNormInputLinear(d_model, 3 * d_model, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
=======
        self.fc1 = LayerNormInputLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
=======
        if isinstance(module, ShiftGaugeEmbedding):
            module.reset_parameters()
        elif isinstance(module, LayerNormInputLinear):
            module.reset_parameters()
        elif isinstance(module, OneColumnGaugeLinear):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.states = [
            {
                "step": 0,
                "exp_avg": torch.zeros_like(reflector),
                "exp_avg_sq": torch.zeros_like(reflector),
            }
            for _, reflector, _ in entries
        ]
=======
        self.states = [
            {
                "step": 0,
                "exp_avg": parameter.new_zeros(
                    (*parameter.shape[:-1], reflector.numel())
                ),
                "exp_avg_sq": parameter.new_zeros(
                    (*parameter.shape[:-1], reflector.numel())
                ),
            }
            for parameter, reflector, _ in entries
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        projection = 2.0 * torch.dot(reflector, value) / norm_sq
        return value - reflector * projection
=======
        projection = 2.0 * (value @ reflector) / norm_sq
        return value - projection.unsqueeze(-1) * reflector
>>>>>>> REPLACE

<<<<<<< SEARCH
            padded = torch.cat((parameter.new_zeros(1), parameter))
            ambient = self._householder(padded, reflector, norm_sq)
            grad_padded = torch.cat(
                (parameter.grad.new_zeros(1), parameter.grad)
            )
=======
            padded = torch.cat(
                (
                    parameter.new_zeros((*parameter.shape[:-1], 1)),
                    parameter,
                ),
                dim=-1,
            )
            ambient = self._householder(padded, reflector, norm_sq)
            grad_padded = torch.cat(
                (
                    parameter.grad.new_zeros(
                        (*parameter.grad.shape[:-1], 1)
                    ),
                    parameter.grad,
                ),
                dim=-1,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    ]
    gauge_ids = {id(parameter) for parameter, _, _ in gauge_entries}
=======
    ] + [
        (
            layer.coordinates,
            layer.reflector,
            layer.reflector_norm_sq,
        )
        for block in model.blocks
        for layer in (block.attn.qkv, block.mlp.fc1)
    ]
    gauge_ids = {id(parameter) for parameter, _, _ in gauge_entries}
>>>>>>> REPLACE