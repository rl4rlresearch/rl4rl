MECHANISM: Lower-triangular first-head value-basis gauge fixing

HYPOTHESIS: The 823-parameter transformer will retain at least 99% accuracy because zeroing the second value row’s third input coefficient removes a remaining within-pivot shear degree while leaving its diagonal scale learnable.

INTENDED_EDIT: Add one zero-fixed off-diagonal coefficient to the first head’s value matrix, reducing learned parameters from 824 to 823.

EVIDENCE: The 824-parameter design reached 99.54% with two first-row shears fixed; prior 823-parameter attempts either constrained a non-pivot coefficient or fixed a diagonal coefficient at 0.02, motivating an untested within-pivot shear constraint that preserves the diagonal scale.

<<<<<<< SEARCH
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and two V shears fixed."""
=======
class GaugeFixedQKV(nn.Module):
    """Quotient-input QKV map with Q/K, V scales, and three V shears fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        selected_indices.update(shear_indices)
=======
        shear_indices = {
            (2 * d_model) * in_features + offset
            for offset in (1, 2)
        }
        shear_indices.add(
            (2 * d_model + 1) * in_features + 2
        )
        selected_indices.update(shear_indices)
>>>>>>> REPLACE