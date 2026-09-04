MECHANISM: Fourth MLP input-row LayerNorm-null gauge quotient

HYPOTHESIS: Centering the first-quarter `fc1` row will reduce the verified 1,579-parameter model to 1,578 parameters while retaining at least 99% accuracy, because its LayerNorm-null input component can be absorbed by that row’s independent bias.

INTENDED_EDIT: Extend `CompactFirstLinearRow` from three centered rows to four, selecting rows 0, 3, 6, and 11 and reconstructing them in their original order.

EVIDENCE: Centering the first, middle, and last `fc1` rows achieved 99.94% accuracy at 1,579 parameters; another separated row has the same independently biased LayerNorm-null gauge, while extending `fc2` to a fourth centered column reached only 97.44%.

<<<<<<< SEARCH
class CompactFirstLinearRow(nn.Module):
    """Linear layer with three nonadjacent LayerNorm input gauges fixed."""
=======
class CompactFirstLinearRow(nn.Module):
    """Linear layer with four separated LayerNorm input gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.middle_row = linear.out_features // 2
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.middle_row],
                linear.weight[self.middle_row + 1 : -1],
            ),
            dim=0,
        )
=======
        self.quarter_row = linear.out_features // 4
        self.middle_row = linear.out_features // 2
        retained_weight = torch.cat(
            (
                linear.weight[1 : self.quarter_row],
                linear.weight[self.quarter_row + 1 : self.middle_row],
                linear.weight[self.middle_row + 1 : -1],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        scaled_rows = (
            linear.weight[[0, self.middle_row, -1]]
            * self._full_ln_weight()
        )
=======
        scaled_rows = (
            linear.weight[[0, self.quarter_row, self.middle_row, -1]]
            * self._full_ln_weight()
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        middle_retained_start = self.middle_row - 1
        full_weight = torch.cat(
            (
                selected_rows[:1],
                self.weight[:middle_retained_start],
                selected_rows[1:2],
                self.weight[middle_retained_start:],
                selected_rows[2:],
            ),
            dim=0,
        )
=======
        quarter_retained_start = self.quarter_row - 1
        middle_retained_start = self.middle_row - 2
        full_weight = torch.cat(
            (
                selected_rows[:1],
                self.weight[:quarter_retained_start],
                selected_rows[1:2],
                self.weight[
                    quarter_retained_start:middle_retained_start
                ],
                selected_rows[2:3],
                self.weight[middle_retained_start:],
                selected_rows[3:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient three nonadjacent independently biased MLP input rows.
=======
        # Fix seven key rows, the four qualified query rows, and two value rows;
        # also quotient four separated independently biased MLP input rows.
>>>>>>> REPLACE