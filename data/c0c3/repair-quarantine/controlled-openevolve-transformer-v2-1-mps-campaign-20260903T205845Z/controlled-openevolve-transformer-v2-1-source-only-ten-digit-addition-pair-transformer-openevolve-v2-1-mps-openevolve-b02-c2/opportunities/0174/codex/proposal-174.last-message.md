MECHANISM: Complete key-row softmax gauge fixing with verified residual quotient

HYPOTHESIS: Adding the verified fifth centered `fc2` column and centering the remaining key row will yield 1,578 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Use the proven five-column residual parameterization, then remove one independent key-weight direction whose contribution is a source-position-independent attention-logit offset.

EVIDENCE: Reference Design 2 achieved 0.998 accuracy with 1,579 parameters and five centered `fc2` columns; the failed sixth-column design reached only 0.8581, so the next 1,578-parameter test uses the remaining exact key-row softmax invariance instead.

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
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    key_start + 2,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
                    second_head_start + 3,
                ]
            ]
            * full_ln_weight
        )
=======
        scaled_key_weight = (
            linear.weight[key_start : self.value_start] * full_ln_weight
        )
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
        retained_query_end = self.key_start - query_weight.size(0)
        full_weight = torch.cat(
            (
                query_weight[:1],
                self.weight[: self.head_dim - 1],
                query_weight[1:],
                self.weight[
                    self.head_dim - 1 : retained_query_end
                ],
                key_weight,
                self.weight[retained_query_end:],
                value_weight,
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactResidualLinear(nn.Module):
    """Linear layer with four weight-column and bias uniform directions fixed."""
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with five weight-column and bias uniform directions fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_columns = (
            linear.weight[:, :4] - linear.weight[:, :4].mean(
                dim=0,
                keepdim=True,
            )
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 4:].detach().clone())
=======
        centered_columns = (
            linear.weight[:, :5] - linear.weight[:, :5].mean(
                dim=0,
                keepdim=True,
            )
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 5:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Use the qualified nonadjacent LayerNorm anchors, fix seven key rows,
        # four qualified query rows, and two value rows; also quotient one
        # independently biased MLP input row.
=======
        # Use the qualified nonadjacent LayerNorm anchors, fix all eight key
        # rows, four qualified query rows, and two value rows; also quotient
        # one MLP input row and five MLP output-column uniform directions.
>>>>>>> REPLACE