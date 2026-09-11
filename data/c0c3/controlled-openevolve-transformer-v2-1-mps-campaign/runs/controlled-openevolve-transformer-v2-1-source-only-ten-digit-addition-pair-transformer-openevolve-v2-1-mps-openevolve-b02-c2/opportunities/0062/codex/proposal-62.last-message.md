MECHANISM: Alternative fourth LayerNorm–key-weight gauge quotient

HYPOTHESIS: Quotienting the third key coordinate of head 0, instead of the failed second coordinate of head 1, will produce a 1,617-parameter model with at least 99% accuracy because it removes the same attention-softmax-invariant direction while avoiding the empirically sensitive head-1 coordinate.

INTENDED_EDIT: Reconstruct four key rows from independent seven-dimensional Helmert coordinates: the first three rows of head 0 and the first row of head 1.

EVIDENCE: Three key-row quotients achieved 99.96% at 1,618 parameters, whereas adding head 1’s second key row fell to 91.96%; testing head 0’s unused third key row isolates whether the failure was coordinate-specific rather than a general four-quotient limit.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with two LayerNorm-induced key-weight gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with four LayerNorm-induced key-weight gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 1 : second_head_start],
                linear.weight[second_head_start + 1 :],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[:key_start],
                linear.weight[key_start + 3 : second_head_start],
                linear.weight[second_head_start + 1 :],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_key_weight = (
            linear.weight[[key_start, second_head_start]] * ln_weight
        )
=======
        scaled_key_weight = (
            linear.weight[
                [key_start, key_start + 1, key_start + 2, second_head_start]
            ]
            * ln_weight
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_retained_start = self.second_key_row - 1
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:1],
                self.weight[self.key_start : second_retained_start],
                key_weight[1:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
=======
        second_retained_start = self.second_key_row - 3
        full_weight = torch.cat(
            (
                self.weight[: self.key_start],
                key_weight[:3],
                self.weight[self.key_start : second_retained_start],
                key_weight[3:],
                self.weight[second_retained_start:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain the qualified bias layout, quotient the effective attention
        # offset, and fix one key-weight gauge in each attention head.
=======
        # Retain the qualified bias layout, quotient the effective attention
        # offset, and fix three head-0 key gauges plus one head-1 key gauge.
>>>>>>> REPLACE