MECHANISM: Single-rotation query/key gauge with function-preserving initialization

HYPOTHESIS: Removing one rotationally redundant query/key parameter will reduce the verified model from 1,272 to 1,271 parameters while retaining at least 99% accuracy, because the chart preserves the dense initialization’s exact attention function and changes only one centered rotation instead of six simultaneously.

INTENDED_EDIT: Replace the tied query/key projection with a one-parameter-smaller chart for the first head, fixing one centered-basis coefficient while leaving the value projection and second head dense.

EVIDENCE: Removing all six query/key rotational redundancies fell to 55.34%, while the 1,272-parameter baseline reached 100%; a single well-conditioned gauge with initialization obtained by rotating an ordinary dense sample isolates whether the earlier failure came from imposing all six constraints at once.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
=======
class SingleRotationGaugeQKV(nn.Module):
    """Tied query/key and value map with one centered rotation fixed."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        in_features = d_model - 1
        head_dim = d_model // n_head
        if head_dim != 4:
            raise ValueError("single-rotation chart requires four-dimensional heads")

        common = torch.full((head_dim,), 1.0 / math.sqrt(head_dim))
        center0 = torch.tensor((1.0, -1.0, 0.0, 0.0)) / math.sqrt(2.0)
        center1 = torch.tensor((1.0, 1.0, -2.0, 0.0)) / math.sqrt(6.0)
        center2 = torch.tensor((1.0, 1.0, 1.0, -3.0)) / math.sqrt(12.0)
        self.register_buffer(
            "head_basis",
            torch.stack((common, center0, center1, center2), dim=1),
            persistent=False,
        )

        self.qk_common = nn.Parameter(torch.empty(1, in_features))
        self.qk_center0 = nn.Parameter(torch.empty(1, in_features))
        self.qk_center1_tail = nn.Parameter(torch.empty(1, in_features - 1))
        self.qk_center2 = nn.Parameter(torch.empty(1, in_features))
        self.qk_rest = nn.Parameter(torch.empty(d_model - head_dim, in_features))
        self.v_weight = nn.Parameter(torch.empty(d_model, in_features))

    def reset_parameters(self) -> None:
        with torch.no_grad():
            dense_head = torch.empty_like(
                self.qk_common.expand(self.head_basis.size(0), -1)
            )
            nn.init.normal_(dense_head, mean=0.0, std=0.02)
            coeff = self.head_basis.transpose(0, 1) @ dense_head

            first = coeff[1].clone()
            second = coeff[2].clone()
            radius = torch.sqrt(first[0].square() + second[0].square())
            radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
            cosine = first[0] / radius
            sine = second[0] / radius
            coeff[1] = cosine * first + sine * second
            coeff[2] = -sine * first + cosine * second

            self.qk_common.copy_(coeff[:1])
            self.qk_center0.copy_(coeff[1:2])
            self.qk_center1_tail.copy_(coeff[2:3, 1:])
            self.qk_center2.copy_(coeff[3:4])
            nn.init.normal_(self.qk_rest, mean=0.0, std=0.02)
            nn.init.normal_(self.v_weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        head_coeff = torch.cat(
            (
                self.qk_common,
                self.qk_center0,
                F.pad(self.qk_center1_tail, (1, 0)),
                self.qk_center2,
            ),
            dim=0,
        )
        first_head = self.head_basis @ head_coeff
        qk_weight = torch.cat((first_head, self.qk_rest), dim=0)
        return F.linear(
            x[..., :-1],
            torch.cat((qk_weight, self.v_weight), dim=0),
        )


class FixedBiasLayerNorm(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = MeanZeroInputLinear(d_model, 2 * d_model, bias=False)
=======
        self.qkv = SingleRotationGaugeQKV(d_model, n_head)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, SingleRotationGaugeQKV):
            module.reset_parameters()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE