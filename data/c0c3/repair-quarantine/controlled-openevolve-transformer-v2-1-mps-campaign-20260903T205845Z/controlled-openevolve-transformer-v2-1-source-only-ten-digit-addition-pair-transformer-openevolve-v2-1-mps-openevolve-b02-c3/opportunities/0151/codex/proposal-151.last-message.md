MECHANISM: Extended cross-head key-feature sharing

HYPOTHESIS: The resulting 670-parameter transformer will retain at least 99% accuracy because it reproduces the qualified 671-parameter design and shares only one additional corresponding key coefficient, extending the same reduction that previously achieved 99.96%.

INTENDED_EDIT: Reproduce Reference Design 2’s fourth fixed MLP output-bias coefficient and first cross-head key tie, then deduplicate the adjacent key coefficient through the same learned source.

EVIDENCE: Reference Design 2 achieved 99.96% with one cross-head key coefficient shared, while alternative 670-parameter reductions to query bias and the remaining MLP output bias collapsed to 45.62% and 19.51%; extending the successful key-sharing mechanism is therefore the most informative next reduction.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""
=======
class GaugeFixedQKV(nn.Module):
    """Gauge-fixed QKV map sharing two learned key features across heads."""
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

        # Reuse the first two coefficients of the final key channel from
        # the first head in the corresponding channel of the second head.
        tied_source_indices = tuple(
            (d_model + head_dim - 1) * in_features + offset
            for offset in range(min(2, in_features))
        )
        tied_indices = tuple(
            (d_model + 2 * head_dim - 1) * in_features + offset
            for offset in range(min(2, in_features))
        )

        self.in_features = in_features
        self.out_features = out_features
        self.fixed_indices = tuple(sorted(selected_indices))
        self.tied_indices = tied_indices
        self.special_indices = tuple(
            sorted(self.fixed_indices + self.tied_indices)
        )
        self.tied_source_coeffs = tuple(
            source_index - sum(
                index < source_index for index in self.fixed_indices
            )
            for source_index in tied_source_indices
        )
        self.coeff = nn.Parameter(
            torch.empty(
                out_features * in_features
                - len(self.fixed_indices)
                - len(self.tied_indices)
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        pieces = []
        learned_start = 0
        full_start = 0
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
        pieces.append(self.coeff[learned_start:])
=======
        pieces = []
        learned_start = 0
        full_start = 0
        fixed_offset = 0
        for special_index in self.special_indices:
            width = special_index - full_start
            pieces.append(
                self.coeff[learned_start : learned_start + width]
            )
            if special_index in self.tied_indices:
                tied_offset = self.tied_indices.index(special_index)
                source_coeff = self.tied_source_coeffs[tied_offset]
                pieces.append(
                    self.coeff[source_coeff : source_coeff + 1]
                )
            else:
                pieces.append(self.fixed_coeff[fixed_offset].view(1))
                fixed_offset += 1
            learned_start += width
            full_start = special_index + 1
        pieces.append(self.coeff[learned_start:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=3
        )
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE