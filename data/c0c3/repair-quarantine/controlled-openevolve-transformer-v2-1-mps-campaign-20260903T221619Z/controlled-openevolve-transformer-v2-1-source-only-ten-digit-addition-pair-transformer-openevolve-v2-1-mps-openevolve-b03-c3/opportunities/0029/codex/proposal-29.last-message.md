MECHANISM: Nonadjacent fifth terminal output-direction gauge

HYPOTHESIS: Gauge-fixing terminal MLP columns 0–3 and 11 will produce a 1,621-parameter model with at least 99% accuracy, showing whether the failed five-column result was specific to column 4 rather than the number of gauge vectors.

INTENDED_EDIT: Preserve the first four qualified gauges, leave the previously destabilizing fifth column fully learned, and gauge-fix the final `fc2` column with the existing ambient-coordinate AdamW treatment.

EVIDENCE: Four adjacent terminal-column gauges achieved 99.96% at 1,622 parameters, while adding column 4 collapsed accuracy to 58.92%; substituting a nonadjacent column is the smallest controlled test of column-specific optimization sensitivity.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and four weight-column output gauges removed."""
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and five selected output gauges removed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
=======
        self.gauge_columns = (0, 1, 2, 3, in_features - 1)
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in self.gauge_columns
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 5)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:])
=======
        for column, stored in zip(
            self.gauge_columns, self.weight_prefix
        ):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
=======
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix[:4], dim=1),
                self.weight_rest,
                full_weight_prefix[4].unsqueeze(1),
            ),
            dim=1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
=======
                for column, stored in zip(
                    module.gauge_columns, module.weight_prefix
                ):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and four terminal-weight gauge vectors.
=======
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and five selected terminal-weight gauge vectors.
>>>>>>> REPLACE