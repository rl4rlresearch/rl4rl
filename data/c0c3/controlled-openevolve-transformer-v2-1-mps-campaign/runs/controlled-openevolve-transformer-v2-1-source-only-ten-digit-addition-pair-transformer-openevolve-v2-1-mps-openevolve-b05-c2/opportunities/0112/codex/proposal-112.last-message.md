MECHANISM: Tenth value–output rotation gauge fixing

HYPOTHESIS: Reproducing the qualified 1494-parameter design and fixing second-head value coordinate 2 through the remaining triangular value–output rotation will yield 1493 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Apply the qualified twelfth query rotation, fixed pre-attention scales, and final-bias anchor, then add `(second_query, 2)` to the value rotations and compensate the attention-output projection at initialization.

EVIDENCE: Reference Design 1 achieved 99.97% accuracy at 1494 parameters after completing the analogous twelfth query–key rotation; the second value head currently has the first five triangular rotations, making its sixth orthogonal rotation the closest symmetry-preserving reduction.

<<<<<<< SEARCH
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
=======
            head_two_value_second_stabilizer_rotation,
            head_two_value_final_rotation,
        ) = self._gauge_fix(full_weight)
        self.register_buffer(
            "initial_first_value_rotation",
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer(
            "initial_head_two_value_second_stabilizer_rotation",
            head_two_value_second_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
=======
        self.register_buffer(
            "initial_head_two_value_second_stabilizer_rotation",
            head_two_value_second_stabilizer_rotation.detach().clone(),
            persistent=False,
        )
        self.register_buffer(
            "initial_head_two_value_final_rotation",
            head_two_value_final_rotation.detach().clone(),
            persistent=False,
        )
        self.first_weight = nn.Parameter(fixed_weight[0, 3:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 2:]
        )
=======
        self.head_two_weight = nn.Parameter(
            fixed_weight[self.second_query, 3:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 2:]
        )
=======
        self.head_two_value_weight = nn.Parameter(
            fixed_weight[value_start + self.second_query, 3:]
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (self.second_query + 1, 1),
            (0, 2),
        )
=======
            (self.second_query + 1, 1),
            (0, 2),
            (self.second_query, 2),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            (self.second_query, 1),
            (self.second_query + 1, 1),
        ):
=======
            (self.second_query, 1),
            (self.second_query + 1, 1),
            (self.second_query, 2),
        ):
>>>>>>> REPLACE

<<<<<<< SEARCH
            value_rotations[7],
            value_rotations[8],
        )
=======
            value_rotations[7],
            value_rotations[8],
            value_rotations[9],
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            head_two_value_second_stabilizer_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
=======
            head_two_value_second_stabilizer_rotation,
            head_two_value_final_rotation,
        ) = self._gauge_fix(full_weight)
        with torch.no_grad():
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 2:]
            )
=======
            self.head_two_weight.copy_(
                fixed_weight[self.second_query, 3:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 2:]
            )
=======
            self.head_two_value_weight.copy_(
                fixed_weight[value_start + self.second_query, 3:]
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.initial_head_two_value_second_stabilizer_rotation.copy_(
                head_two_value_second_stabilizer_rotation
            )

    @property
=======
            self.initial_head_two_value_second_stabilizer_rotation.copy_(
                head_two_value_second_stabilizer_rotation
            )
            self.initial_head_two_value_final_rotation.copy_(
                head_two_value_final_rotation
            )

    @property
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_row = F.pad(self.head_two_weight, (2, 0))
=======
        head_two_row = F.pad(self.head_two_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
        head_two_value_row = F.pad(self.head_two_value_weight, (2, 0))
=======
        head_two_value_row = F.pad(self.head_two_value_weight, (3, 0))
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.proj.weight[
                :, head_two_start + 1:head_two_start + 3
            ].copy_(head_two_second_stabilized_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
            self.proj.weight[
                :, head_two_start + 1:head_two_start + 3
            ].copy_(head_two_second_stabilized_columns)
            head_two_final_columns = (
                self.proj.weight[
                    :, head_two_start:head_two_start + 2
                ]
                @ self.qkv.initial_head_two_value_final_rotation.transpose(
                    0, 1
                )
            )
            self.proj.weight[
                :, head_two_start:head_two_start + 2
            ].copy_(head_two_final_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
class SeptupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with seven scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 7), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)


class Block(nn.Module):
=======
class FixedScaleLayerNorm(nn.Module):
    """LayerNorm with all scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.dim,), None, None, self.eps)


class FirstBiasAnchoredLayerNorm(nn.Module):
    """LayerNorm with final bias coordinate 0 fixed at zero."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (1, 0))
        return F.layer_norm(
            x, (self.dim,), self.weight, bias, self.eps
        )


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SeptupleAnchoredScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = FixedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = FirstBiasAnchoredLayerNorm(cfg.d_model)
>>>>>>> REPLACE