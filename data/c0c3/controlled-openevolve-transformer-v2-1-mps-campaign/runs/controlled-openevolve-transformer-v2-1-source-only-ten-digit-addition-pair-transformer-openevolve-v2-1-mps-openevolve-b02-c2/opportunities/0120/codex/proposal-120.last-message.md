MECHANISM: Seventh causal key-offset gauge quotient

HYPOTHESIS: Reproducing the verified 1,583-parameter design and centering the remaining third key row of the first attention head will yield 1,582 parameters with at least 99% accuracy, because the removed component contributes only a position-independent key offset canceled by causal softmax.

INTENDED_EDIT: Apply the qualified positional-row and nonadjacent `ln1` scale reductions, reproduce the six centered key rows, then move key row `key_start + 2` into the centered basis.

EVIDENCE: Six centered key rows achieved 99.92% accuracy at 1,583 parameters, after the analogous fifth and sixth key-row reductions achieved 99.98% and 99.92%; the remaining first-head row has the same softmax-invariant gauge.

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
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and two value-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with seven key, four query, and two value-row gauges fixed."""
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
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[key_start + 2 : second_head_start],
                linear.weight[second_head_start + 2 : -2],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[
                    second_head_start + 2 : second_head_start + 3
                ],
                linear.weight[second_head_start + 4 : -2],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        full_ln_weight = self._full_ln_weight()
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    key_start + 2,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
                    second_head_start + 3,
                ]
            ]
            * full_ln_weight
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_query_weight = (
            linear.weight[
                [
                    0,
                    self.head_dim,
                    self.head_dim + 1,
                    self.head_dim + 2,
                ]
            ]
            * ln_weight
        )
=======
        scaled_query_weight = (
            linear.weight[
                [
                    0,
                    self.head_dim,
                    self.head_dim + 1,
                    self.head_dim + 2,
                ]
            ]
            * full_ln_weight
        )
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
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / self.ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / self.ln_weight
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
        scaled_key_weight = self.key_weight @ self.key_basis.transpose(0, 1)
        key_weight = scaled_key_weight / full_ln_weight
        scaled_value_weight = (
            self.value_weight @ self.key_basis.transpose(0, 1)
        )
        value_weight = scaled_value_weight / full_ln_weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_key_retained_start = self.key_start - 4
        second_key_retained_start = self.second_key_row - 6
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[
                    self.head_dim - 1 : first_key_retained_start
                ],
                key_weight[:2],
                self.weight[
                    first_key_retained_start : second_key_retained_start
                ],
                key_weight[2:],
                self.weight[second_key_retained_start:],
                value_weight,
            ),
            dim=0,
        )
=======
        second_key_retained_start = self.key_start - 4
        second_key_retained_end = second_key_retained_start + 1
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[
                    self.head_dim - 1 : second_key_retained_start
                ],
                key_weight[: self.head_dim],
                key_weight[self.head_dim : self.head_dim + 2],
                self.weight[
                    second_key_retained_start : second_key_retained_end
                ],
                key_weight[self.head_dim + 2 :],
                self.weight[second_key_retained_end:],
                value_weight,
            ),
            dim=0,
        )
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
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 3.
=======
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 4.
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias and projection layout, fix four key rows,
        # four qualified query rows, and two value rows; also quotient one
        # independently biased MLP input row.
=======
        # Use the qualified nonadjacent LayerNorm anchors, fix seven key rows,
        # four qualified query rows, and two value rows; also quotient one
        # independently biased MLP input row.
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