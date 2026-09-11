MECHANISM: Nonadjacent pre-MLP LayerNorm scale absorption

HYPOTHESIS: Centering positional rows 1–6 and fixing a separated middle `ln2` scale coordinate will produce a 1,576-parameter model with at least 99% accuracy, because the verified positional reduction retained 99.81% accuracy and the downstream learned `fc1` weights absorb the removed LayerNorm scale.

INTENDED_EDIT: Reproduce the verified sixth positional-row centering, then anchor `ln2` scale coordinates 0, 4, and 7 while reconstructing the full scale for the compact `fc1` row.

EVIDENCE: Reference Design 1 achieved 99.81% accuracy at 1,577 parameters with positional rows 1–6 centered; supplied evidence also reports that separated LayerNorm anchors succeeded where adjacent extensions failed, motivating a nonadjacent middle scale anchor instead of retrying the repeatedly timed-out full-bias removal.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and six row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and seven row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_positions = embedding.weight[1:6] - embedding.weight[1:6].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[6:].detach().clone())
=======
        centered_positions = embedding.weight[1:7] - embedding.weight[1:7].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[7:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        if fixed_weight_coordinates:
            leading_fixed = fixed_weight_coordinates - 1
            self.weight = nn.Parameter(
                layer_norm.weight[leading_fixed:-1].detach().clone()
            )
        else:
            self.weight = layer_norm.weight
=======
        if fixed_weight_coordinates == 3:
            middle_coordinate = layer_norm.normalized_shape[0] // 2
            retained_weight = torch.cat(
                (
                    layer_norm.weight[1:middle_coordinate],
                    layer_norm.weight[middle_coordinate + 1 : -1],
                )
            )
            self.weight = nn.Parameter(retained_weight.detach().clone())
        elif fixed_weight_coordinates:
            leading_fixed = fixed_weight_coordinates - 1
            self.weight = nn.Parameter(
                layer_norm.weight[leading_fixed:-1].detach().clone()
            )
        else:
            self.weight = layer_norm.weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        if self.fixed_weight_coordinates:
            leading_fixed = self.fixed_weight_coordinates - 1
            full_weight = torch.cat(
                (
                    self.weight.new_ones(leading_fixed),
                    self.weight,
                    self.weight.new_ones(1),
                )
            )
        else:
            full_weight = self.weight
=======
        if self.fixed_weight_coordinates == 3:
            leading_retained = (self.weight.numel() + 3) // 2 - 1
            full_weight = torch.cat(
                (
                    self.weight.new_ones(1),
                    self.weight[:leading_retained],
                    self.weight.new_ones(1),
                    self.weight[leading_retained:],
                    self.weight.new_ones(1),
                )
            )
        elif self.fixed_weight_coordinates:
            leading_fixed = self.fixed_weight_coordinates - 1
            full_weight = torch.cat(
                (
                    self.weight.new_ones(leading_fixed),
                    self.weight,
                    self.weight.new_ones(1),
                )
            )
        else:
            full_weight = self.weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
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
=======
        self.row_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_row).detach().clone()
        )

    def _full_ln_weight(self) -> torch.Tensor:
        if self.fixed_ln_weight_coordinates == 3:
            leading_retained = (self.ln_weight.numel() + 3) // 2 - 1
            return torch.cat(
                (
                    self.ln_weight.new_ones(1),
                    self.ln_weight[:leading_retained],
                    self.ln_weight.new_ones(1),
                    self.ln_weight[leading_retained:],
                    self.ln_weight.new_ones(1),
                )
            )
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=7,
            fixed_weight_coordinates=2,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=7,
            fixed_weight_coordinates=3,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 4.
=======
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 6.
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Use the qualified nonadjacent LayerNorm anchors, fix seven key rows,
        # four qualified query rows, and two value rows; also quotient one
        # independently biased MLP input row.
=======
        # Use qualified nonadjacent LayerNorm anchors, including a middle ln2
        # scale anchor; fix seven key rows, four query rows, two value rows,
        # and one independently biased MLP input row.
>>>>>>> REPLACE