MECHANISM: Nonadjacent downstream LayerNorm scale gauge quotient

HYPOTHESIS: Fixing the first and last `ln2` scale coordinates will yield 1,588 parameters with at least 99% accuracy because `fc1` can absorb both scales, while retaining trainable bias on the newly fixed leading coordinate may avoid the conditioning failure of fixing the last two adjacent scales.

INTENDED_EDIT: Preserve the verified final-coordinate scale anchor, add a leading-coordinate anchor, and reconstruct the full scale consistently in `ln2` and the compact first MLP row.

EVIDENCE: Fixing one trailing `ln2` scale achieved 99.96% accuracy at 1,589 parameters, while fixing the two trailing scales reached 98.79%; selecting a nonadjacent second coordinate with trainable `ln2` bias tests whether that failure was specific to the adjacent fully bias-fixed coordinates.

<<<<<<< SEARCH
        if fixed_weight_coordinates:
            self.weight = nn.Parameter(
                layer_norm.weight[:-fixed_weight_coordinates].detach().clone()
            )
        else:
            self.weight = layer_norm.weight
=======
        if fixed_weight_coordinates:
            leading_fixed = fixed_weight_coordinates - 1
            self.weight = nn.Parameter(
                layer_norm.weight[leading_fixed:-1].detach().clone()
            )
        else:
            self.weight = layer_norm.weight
>>>>>>> REPLACE

<<<<<<< SEARCH
        if self.fixed_weight_coordinates:
            full_weight = torch.cat(
                (
                    self.weight,
                    self.weight.new_ones(self.fixed_weight_coordinates),
                )
            )
        else:
            full_weight = self.weight
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _full_ln_weight(self) -> torch.Tensor:
        if self.fixed_ln_weight_coordinates:
            return torch.cat(
                (
                    self.ln_weight,
                    self.ln_weight.new_ones(
                        self.fixed_ln_weight_coordinates
                    ),
                )
            )
        return self.ln_weight
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=1,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE