MECHANISM: Causal-softmax key-offset and final-LayerNorm residual gauges

HYPOTHESIS: Centering seven key rows and three `fc2` output columns will reduce the model from 1,584 to 1,581 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the verified 1,581-parameter construction by adding two centered key rows and one centered residual-output column.

EVIDENCE: Reference Design 2 achieved 0.9987 accuracy with 1,581 parameters after 4,999 training steps using this exact construction.

<<<<<<< SEARCH
class CompactQKV(nn.Module):
    """Compact QKV with five key, four query, and two value-row gauges fixed."""
=======
class CompactQKV(nn.Module):
    """Compact QKV with seven key, four query, and two value-row gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.head_dim],
                linear.weight[self.head_dim + 3 : key_start],
                linear.weight[key_start + 2 : second_head_start - 1],
                linear.weight[second_head_start + 2 : -2],
            ),
            dim=0,
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_key_weight = (
            linear.weight[
                [
                    key_start,
                    key_start + 1,
                    second_head_start - 1,
                    second_head_start,
                    second_head_start + 1,
                ]
            ]
            * full_ln_weight
        )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        first_key_retained_start = self.key_start - 4
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
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class CompactResidualLinear(nn.Module):
    """Linear layer with two weight-column and bias uniform directions fixed."""
=======
class CompactResidualLinear(nn.Module):
    """Linear layer with three weight-column and bias uniform directions fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_columns = (
            linear.weight[:, :2]
            - linear.weight[:, :2].mean(dim=0, keepdim=True)
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 2:].detach().clone())
=======
        centered_columns = (
            linear.weight[:, :3] - linear.weight[:, :3].mean(
                dim=0,
                keepdim=True,
            )
        )
        self.column_weight = nn.Parameter(
            (basis.transpose(0, 1) @ centered_columns).detach().clone()
        )
        self.weight = nn.Parameter(linear.weight[:, 3:].detach().clone())
>>>>>>> REPLACE