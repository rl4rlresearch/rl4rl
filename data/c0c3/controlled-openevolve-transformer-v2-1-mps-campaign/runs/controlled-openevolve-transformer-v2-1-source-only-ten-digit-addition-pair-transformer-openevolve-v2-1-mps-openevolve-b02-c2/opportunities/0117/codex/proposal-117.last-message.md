MECHANISM: Nonadjacent fifth key-row LayerNorm-null gauge

HYPOTHESIS: Centering the final key row of the first attention head will reduce the verified 1,585-parameter model to 1,584 parameters while retaining at least 99% accuracy, because its removed uniform component produces only a position-independent key offset that causal softmax cancels.

INTENDED_EDIT: Move key row `second_head_start - 1` from the full QKV weight into the seven-dimensional centered key basis and reconstruct it in its original order.

EVIDENCE: The current model achieved 99.96% accuracy with four centered key rows. Successful nonadjacent LayerNorm anchors outperformed adjacent extensions, motivating a separated final-row anchor instead of centering the adjacent third key row.

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
                [
                    key_start,
                    key_start + 1,
                    second_head_start,
                    second_head_start + 1,
                ]
=======
                [
                    key_start,
                    key_start + 1,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
                ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_key_retained_start = self.key_start - 4
        second_key_retained_start = self.second_key_row - 6
=======
        first_key_retained_start = self.key_start - 4
        second_key_retained_start = self.second_key_row - 7
>>>>>>> REPLACE

<<<<<<< SEARCH
                key_weight[:2],
                self.weight[
                    first_key_retained_start : second_key_retained_start
                ],
                key_weight[2:],
                self.weight[second_key_retained_start:],
=======
                key_weight[:2],
                self.weight[
                    first_key_retained_start : second_key_retained_start
                ],
                key_weight[2:3],
                key_weight[3:],
                self.weight[second_key_retained_start:],
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix four key rows, the four qualified query rows, and two value rows;
=======
        # Fix five key rows, the four qualified query rows, and two value rows;
>>>>>>> REPLACE