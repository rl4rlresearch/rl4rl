MECHANISM: Cross-head key-feature sharing

HYPOTHESIS: Sharing one corresponding key-projection coefficient across attention heads will produce a 671-parameter transformer while retaining at least 99% accuracy, because head-specific queries and relative routing remain fully learned.

INTENDED_EDIT: Challenge the assumption that both heads require completely independent key projections by deduplicating one second-head key coefficient and reusing its first-head counterpart.

EVIDENCE: The current 672-parameter model reaches 99.52%, while removing another query-bias coordinate collapsed accuracy to 45.62%, indicating query biases are load-bearing. Qualified designs already tolerate identical fixed coefficients in corresponding key channels, motivating a less disruptive learned key sharing constraint.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""
=======
class GaugeFixedQKV(nn.Module):
    """Gauge-fixed QKV map sharing one learned key feature across heads."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        selected_indices.update(shear_indices)

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.coeff = nn.Parameter(
            torch.empty(out_features * in_features - len(self.fixed_indices))
        )
=======
        selected_indices.update(shear_indices)

        # Reuse the first coefficient of the final key channel from the
        # first head in the corresponding channel of the second head.
        # Queries and all other key coordinates remain head-specific.
        tied_source_index = (
            d_model + head_dim - 1
        ) * in_features
        tied_index = (
            d_model + 2 * head_dim - 1
        ) * in_features

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.tied_index = tied_index
        self.special_indices = tuple(
            sorted(self.fixed_indices + (self.tied_index,))
        )
        self.tied_source_coeff = tied_source_index - sum(
            index < tied_source_index for index in self.fixed_indices
        )
        self.coeff = nn.Parameter(
            torch.empty(
                out_features * in_features
                - len(self.fixed_indices)
                - 1
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for fixed_index, fixed_coeff in zip(
            self.fixed_indices, self.fixed_coeff
        ):
            width = fixed_index - full_start
            pieces.append(
                self.coeff[learned_start : learned_start + width]
            )
            pieces.append(fixed_coeff.view(1))
            learned_start += width
            full_start = fixed_index + 1
=======
        fixed_offset = 0
        for special_index in self.special_indices:
            width = special_index - full_start
            pieces.append(
                self.coeff[learned_start : learned_start + width]
            )
            if special_index == self.tied_index:
                pieces.append(
                    self.coeff[
                        self.tied_source_coeff : self.tied_source_coeff + 1
                    ]
                )
            else:
                pieces.append(self.fixed_coeff[fixed_offset].view(1))
                fixed_offset += 1
            learned_start += width
            full_start = special_index + 1
>>>>>>> REPLACE