MECHANISM: Within-head second key-row normalized-input gauge

HYPOTHESIS: Extending the verified 1,582-parameter design with key row 9 will produce a 1,581-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the four balanced query-row gauges and key-row-8 gauge, then omit the final normalized-input coefficient from adjacent key row 9.

EVIDENCE: Key row 8 achieved 99.93% accuracy at 1,582 parameters, while adding second-head key row 12 fell to 66.0%; row 9 tests whether the first head can tolerate another key gauge without constraining the sensitive second head.

<<<<<<< SEARCH
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0,)
        self.ungauged_rows = tuple(range(1, self.out_features))
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        # Balance two query gauges per head and extend the successful first-
        # head key gauge to its adjacent key coordinate.
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            d_model + 1,
        )
        self.ungauged_rows = tuple(
            row for row in range(self.out_features)
            if row not in self.gauged_rows
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        row_width = self.in_features - 1
        gauged = torch.cat(
            (
                self.weight[:row_width].view(1, row_width),
                self.weight.new_zeros(1, 1),
            ),
            dim=1,
        )
        ungauged = self.weight[row_width:].view(
            len(self.ungauged_rows), self.in_features
        )
        weight = torch.cat((gauged, ungauged), dim=0)
=======
        row_width = self.in_features - 1
        split = len(self.gauged_rows) * row_width
        gauged = torch.cat(
            (
                self.weight[:split].view(len(self.gauged_rows), row_width),
                self.weight.new_zeros(len(self.gauged_rows), 1),
            ),
            dim=1,
        )
        ungauged = self.weight[split:].view(
            len(self.ungauged_rows), self.in_features
        )

        rows = []
        gauged_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.gauged_rows:
                rows.append(gauged[gauged_index])
                gauged_index += 1
            else:
                rows.append(ungauged[ungauged_index])
                ungauged_index += 1
        weight = torch.stack(rows)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias and one representative of QKV row 0's normalized-
        # input coefficient gauge.
        self.qkv = LayerNormGaugedQKV(d_model)
=======
        # Retain query bias while gauge-fixing balanced query rows and the
        # first two key coordinates of the first attention head.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # All attention scales initialize to one, so subtracting the
                # omitted coefficient preserves the initial row function.
=======
                # All attention scales initialize to one, so subtracting each
                # omitted coefficient preserves the selected row functions.
>>>>>>> REPLACE