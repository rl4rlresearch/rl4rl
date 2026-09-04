MECHANISM: Attention-input LayerNorm scale gauge quotient

HYPOTHESIS: Extending the verified 1,587-parameter positional-row design with one fixed trailing `ln1` scale will yield 1,586 parameters and at least 99% accuracy, because all downstream QKV columns can absorb that scale.

INTENDED_EDIT: Zero-center positional rows 1–4, fix the final `ln1` scale coordinate at one, and reconstruct its full scale consistently inside LayerNorm and compact QKV.

EVIDENCE: Centering positional row 4 achieved 99.84% accuracy at 1,587 parameters, while the analogous single trailing `ln2` scale anchor achieved 99.96%; this tests the same downstream-linear redundancy on the independently parameterized attention branch.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and four row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and five row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_positions = embedding.weight[1:4] - embedding.weight[1:4].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[4:].detach().clone())
=======
        centered_positions = embedding.weight[1:5] - embedding.weight[1:5].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[5:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_head_start: int,
        ln_weight: nn.Parameter,
    ):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.second_key_row = second_head_start
        self.ln_weight = ln_weight
=======
        second_head_start: int,
        ln_weight: nn.Parameter,
        fixed_ln_weight_coordinates: int = 0,
    ):
        super().__init__()
        self.key_start = key_start
        self.value_start = 2 * key_start
        self.head_dim = second_head_start - key_start
        self.second_key_row = second_head_start
        self.ln_weight = ln_weight
        self.fixed_ln_weight_coordinates = fixed_ln_weight_coordinates
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("key_basis", basis, persistent=False)

        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * ln_weight
        )
=======
        self.register_buffer("key_basis", basis, persistent=False)

        full_ln_weight = self._full_ln_weight()
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * full_ln_weight
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            ]
            * ln_weight
        )
        centered_query_weight = (
=======
            ]
            * full_ln_weight
        )
        centered_query_weight = (
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_value_weight = linear.weight[-2:] * ln_weight
=======
        scaled_value_weight = linear.weight[-2:] * full_ln_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaled_query_weight = (
            self.query_weight @ self.key_basis.transpose(0, 1)
        )
        query_weight = scaled_query_weight / self.ln_weight
=======
        self.value_bias = nn.Parameter(
            linear.bias[self.value_start + self.head_dim + 3 :].detach().clone()
        )

    def _full_ln_weight(self) -> torch.Tensor:
        if self.fixed_ln_weight_coordinates:
            leading_fixed = self.fixed_ln_weight_coordinates - 1
            return torch.cat(
                (
                    self.ln_weight.new_ones(leading_fixed),
                    self.ln_weight,
                    self.ln_weight.new_ones(1),
                )
            )
        return self.ln_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_ln_weight = self._full_ln_weight()
        scaled_query_weight = (
            self.query_weight @ self.key_basis.transpose(0, 1)
        )
        query_weight = scaled_query_weight / full_ln_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_weight = scaled_key_weight / self.ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / self.ln_weight
=======
        key_weight = scaled_key_weight / full_ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / full_ln_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
        )
=======
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
            fixed_weight_coordinates=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
            )
=======
                cfg.d_model + block.attn.head_dim,
                block.ln1.weight,
                block.ln1.fixed_weight_coordinates,
            )
>>>>>>> REPLACE