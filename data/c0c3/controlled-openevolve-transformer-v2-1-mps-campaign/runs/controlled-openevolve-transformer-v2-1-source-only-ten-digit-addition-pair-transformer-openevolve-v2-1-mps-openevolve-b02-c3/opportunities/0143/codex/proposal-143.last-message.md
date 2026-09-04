MECHANISM: First-head Q/K stabilizer shear gauge fixing

HYPOTHESIS: A 661-parameter transformer will retain at least 99% accuracy because fixing one first-head key coefficient removes a remaining exact Q/K shear redundancy without reducing representable attention scores.

INTENDED_EDIT: Fix the second input coefficient of the first head’s first key channel at zero, reducing the learned QKV parameter count by one.

EVIDENCE: The current 662-parameter model achieved 99.88% after per-head Q/K scale fixing; with all three first-head key rows sharing the same fixed first coefficient, a determinant-one row shear preserves those constraints and can absorb this additional coefficient.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K scales and shears and V gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        selected_indices.update(shear_indices)
=======
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        # Since the first coefficients of all three first-head key rows are
        # equal, adding a multiple of row 1 minus row 2 to row 0 preserves
        # them while allowing this coefficient to be fixed at zero.
        shear_indices.add(d_model * in_features + 1)
        selected_indices.update(shear_indices)
>>>>>>> REPLACE