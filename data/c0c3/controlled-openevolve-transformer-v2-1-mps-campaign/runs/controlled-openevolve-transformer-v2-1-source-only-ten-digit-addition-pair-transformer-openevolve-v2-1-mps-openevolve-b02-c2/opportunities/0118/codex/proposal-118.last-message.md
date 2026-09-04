MECHANISM: Nonadjacent attention-scale anchor plus causal key-offset gauge quotient

HYPOTHESIS: Reproducing the verified 1,584-parameter design will retain at least 99% accuracy because the added leading `ln1` scale anchor is absorbed by QKV and centering the separated final row of the first key head removes only a softmax-invariant key offset.

INTENDED_EDIT: Fix the first and last `ln1` scale coordinates and move key row `second_head_start - 1` into the centered key basis, reducing the model from 1,586 to 1,584 learned parameters.

EVIDENCE: Reference Design 1 used this exact combination and achieved 99.98% accuracy with 1,584 parameters; the current design already contains all of its other qualified compactions.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with four key, four query, and two value-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with five key, four query, and two value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                linear.weight[key_start + 2 : second_head_start],
=======
                linear.weight[key_start + 2 : second_head_start - 1],
>>>>>>> REPLACE

<<<<<<< SEARCH
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
=======
                    key_start,
                    key_start + 1,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        second_key_retained_start = self.second_key_row - 7
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
                key_weight[2:3],
                key_weight[3:],
                self.weight[second_key_retained_start:],
                value_weight,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=2,
            fixed_weight_coordinates=1,
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
        # Fix four key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
=======
        # Fix five key rows, the four qualified query rows, and two value rows;
        # also quotient one independently biased MLP input row.
>>>>>>> REPLACE