MECHANISM: Narrow-content positional attention

HYPOTHESIS: Reducing each attention head’s learned query/key width from four dimensions to two will produce a 895-parameter transformer with at least 99% accuracy, because learned relative-lag biases can perform most positional routing while the unchanged four-dimensional value paths, token representation, and MLP retain the task’s load-bearing capacity.

INTENDED_EDIT: Replace the 24-output QKV projection with a 16-output projection containing two-dimensional queries and keys plus full-width values, preserving the conceptual initialization stream and input-shift gauge optimization.

EVIDENCE: The four-channel token bottleneck retained 99.84% accuracy while three channels reached only 96.91%, and tying the MLP readout collapsed accuracy to 37.01%; this motivates preserving token, value, and MLP capacity while testing whether full-width content addressing is unnecessary alongside the learned relative-lag table.

<<<<<<< SEARCH
class OneRowInputGaugeLinear(nn.Module):
=======
class NarrowQKVLinear(nn.Module):
    """Gauge-fixed Q/K bottleneck with an unchanged full-width value path."""

    def __init__(
        self, d_model: int, n_head: int, qk_dim: int
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qk_dim = qk_dim
        self.out_features = 2 * n_head * qk_dim + d_model
        self.coordinates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(d_model - 1))
                for _ in range(self.out_features)
            ]
        )

        inv_sqrt = d_model ** -0.5
        reflector = torch.full((d_model,), -inv_sqrt)
        reflector[0] += 1.0
        self.register_buffer("reflector", reflector, persistent=False)
        self.reflector_norm_sq = float(reflector.dot(reflector))

        # Consume the original Linear(8, 24) constructor draw. Selecting the
        # leading Q/K coordinates of each head also preserves their conceptual
        # initialization, while all value coordinates remain unchanged.
        conceptual_weight = torch.empty(3 * d_model, d_model)
        nn.init.kaiming_uniform_(conceptual_weight, a=math.sqrt(5))
        self._set_weight(conceptual_weight)

    def _householder(self, value: torch.Tensor) -> torch.Tensor:
        projection = (
            2.0 * (value @ self.reflector) / self.reflector_norm_sq
        )
        return value - projection.unsqueeze(-1) * self.reflector

    def _select_weight(
        self, conceptual_weight: torch.Tensor
    ) -> torch.Tensor:
        q = conceptual_weight[: self.d_model].view(
            self.n_head, self.head_dim, self.d_model
        )
        k = conceptual_weight[
            self.d_model : 2 * self.d_model
        ].view(self.n_head, self.head_dim, self.d_model)
        q = q[:, : self.qk_dim, :].reshape(-1, self.d_model)
        k = k[:, : self.qk_dim, :].reshape(-1, self.d_model)
        v = conceptual_weight[2 * self.d_model :]
        return torch.cat((q, k, v), dim=0)

    def _set_weight(self, conceptual_weight: torch.Tensor) -> None:
        with torch.no_grad():
            transformed = self._householder(
                self._select_weight(conceptual_weight)
            )
            for coordinates, row in zip(self.coordinates, transformed):
                coordinates.copy_(row[1:])

    def reset_parameters(self) -> None:
        conceptual_weight = self.coordinates[0].new_empty(
            3 * self.d_model, self.d_model
        )
        nn.init.normal_(conceptual_weight, mean=0.0, std=0.02)
        self._set_weight(conceptual_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coordinates = torch.stack(list(self.coordinates), dim=0)
        weight = self._householder(F.pad(coordinates, (1, 0)))
        return F.linear(x, weight)


class OneRowInputGaugeLinear(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = OneRowInputGaugeLinear(d_model, 3 * d_model)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qk_dim = self.head_dim // 2
        if self.qk_dim < 1:
            raise ValueError("attention heads must have at least two channels")
        self.qkv = NarrowQKVLinear(d_model, n_head, self.qk_dim)
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + F.pad(self.q_bias, (0, 6))
        v = v + F.pad(self.v_bias, (0, 5))

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
=======
        qkv = self.qkv(x)
        qk_width = self.n_head * self.qk_dim
        q, k, v = torch.split(
            qkv, (qk_width, qk_width, d_model), dim=-1
        )
        q = q + F.pad(
            self.q_bias, (0, qk_width - self.q_bias.numel())
        )
        v = v + F.pad(self.v_bias, (0, 5))

        q = q.view(
            bsz, seqlen, self.n_head, self.qk_dim
        ).transpose(1, 2)
        k = k.view(
            bsz, seqlen, self.n_head, self.qk_dim
        ).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.qk_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, OneRowInputGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, OneRowInputGaugeLinear):
            module.reset_parameters()
        elif isinstance(module, NarrowQKVLinear):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
    ] + [
        (
            coordinates,
            block.attn.qkv.reflector,
            block.attn.qkv.reflector_norm_sq,
        )
        for block in model.blocks
        for coordinates in (
            block.attn.qkv.first_coordinates,
            block.attn.qkv.second_coordinates,
            block.attn.qkv.third_coordinates,
            block.attn.qkv.fourth_coordinates,
            block.attn.qkv.fifth_coordinates,
            block.attn.qkv.sixth_coordinates,
            block.attn.qkv.seventh_coordinates,
            block.attn.qkv.eighth_coordinates,
            block.attn.qkv.ninth_coordinates,
            block.attn.qkv.tenth_coordinates,
            block.attn.qkv.eleventh_coordinates,
            block.attn.qkv.twelfth_coordinates,
            block.attn.qkv.thirteenth_coordinates,
            block.attn.qkv.fourteenth_coordinates,
            block.attn.qkv.fifteenth_coordinates,
            block.attn.qkv.sixteenth_coordinates,
            block.attn.qkv.seventeenth_coordinates,
            block.attn.qkv.eighteenth_coordinates,
            block.attn.qkv.nineteenth_coordinates,
        )
    ]
=======
    ] + [
        (
            coordinates,
            block.attn.qkv.reflector,
            block.attn.qkv.reflector_norm_sq,
        )
        for block in model.blocks
        for coordinates in block.attn.qkv.coordinates
    ]
>>>>>>> REPLACE