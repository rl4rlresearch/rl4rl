MECHANISM: Eighth causal key-offset gauge quotient

HYPOTHESIS: Centering the remaining third key row of the second attention head will reduce the verified 1,582-parameter model to 1,581 parameters while retaining at least 99% accuracy, because the removed component produces only a position-independent key offset canceled by causal softmax.

INTENDED_EDIT: Move key row `second_head_start + 2` from the full QKV weight into the centered key basis and reconstruct all eight key rows in their original order.

EVIDENCE: Seven centered key rows achieved 99.91% accuracy at 1,582 parameters, and the fifth, sixth, and seventh key-row reductions all remained above 99%; the sole remaining key row has the same softmax-invariant gauge.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with seven key, four query, and two value-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with eight key, four query, and two value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[
                    second_head_start + 2 : second_head_start + 3
                ],
                linear.weight[second_head_start + 4 : -2],
            ),
            dim=0,
        )
=======
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[second_head_start + 4 : -2],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                    second_head_start,
                    second_head_start + 1,
                    second_head_start + 3,
=======
                    second_head_start,
                    second_head_start + 1,
                    second_head_start + 2,
                    second_head_start + 3,
>>>>>>> REPLACE

<<<<<<< SEARCH
        second_key_retained_start = self.key_start - 4
        second_key_retained_end = second_key_retained_start + 1
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[
                    self.head_dim - 1 : second_key_retained_start
                ],
                key_weight[: self.head_dim],
                key_weight[self.head_dim : self.head_dim + 2],
                self.weight[
                    second_key_retained_start : second_key_retained_end
                ],
                key_weight[self.head_dim + 2 :],
                self.weight[second_key_retained_end:],
                value_weight,
            ),
            dim=0,
        )
=======
        retained_query_end = self.key_start - 4
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[self.head_dim - 1 : retained_query_end],
                key_weight,
                self.weight[retained_query_end:],
                value_weight,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Use the qualified nonadjacent LayerNorm anchors, fix seven key rows,
        # four qualified query rows, and two value rows; also quotient one
=======
        # Use the qualified nonadjacent LayerNorm anchors, fix all eight key rows,
        # four qualified query rows, and two value rows; also quotient one
>>>>>>> REPLACE