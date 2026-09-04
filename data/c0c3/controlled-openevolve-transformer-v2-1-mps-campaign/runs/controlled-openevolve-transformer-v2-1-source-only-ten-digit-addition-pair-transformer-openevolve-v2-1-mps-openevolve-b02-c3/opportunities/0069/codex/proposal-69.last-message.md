MECHANISM: Second first-head value-space shear gauge fixing

HYPOTHESIS: An 824-parameter transformer will retain at least 99% accuracy because fixing a second off-axis coefficient in the first value channel removes another V/output-projection shear degree without reducing attention routing or representational width.

INTENDED_EDIT: Fix the third input coefficient of the first value channel at zero, extending the qualified first-head value gauge while leaving the sensitive second head unchanged.

EVIDENCE: The current 825-parameter design achieved 99.99% accuracy after fixing the adjacent second input coefficient of the same first-head value channel; prior constraints involving the second head collapsed, so the next informative reduction remains within the demonstrated tolerant first head.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and one V shear fixed."""
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and two V shears fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        shear_index = (2 * d_model) * in_features + 1
        selected_indices.add(shear_index)
=======
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        selected_indices.update(shear_indices)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    0.0 if index == shear_index else 0.02
=======
                    0.0 if index in shear_indices else 0.02
>>>>>>> REPLACE