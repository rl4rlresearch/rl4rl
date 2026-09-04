MECHANISM: Alternate query-bias coordinate ablation

HYPOTHESIS: Retaining query-bias coordinates 1–3 while fixing coordinate 0 will produce 1600 parameters and maintain at least 99% accuracy, indicating that the previous three-coordinate failure was caused by removing load-bearing coordinate 3 rather than by insufficient bias dimensionality.

INTENDED_EDIT: Replace the four-coordinate query bias with three learned coordinates occupying indices 1–3, preserving all weight tensors, initialization streams, positional quotient updates, and training behavior.

EVIDENCE: Four query-bias coordinates achieved 99.96% at 1601 parameters, while learning only coordinates 0–2 fell to 73.19%; testing the complementary removal within the same proven four-coordinate head directly distinguishes a coordinate-specific failure from a three-parameter capacity limit.

<<<<<<< SEARCH
        # Construct the baseline layer first to preserve the proven constructor
        # RNG stream. Key and value biases are omitted, and only the first four
        # query coordinates are learned; all weights retain baseline shapes.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
=======
        # Construct the baseline layer first to preserve the proven constructor
        # RNG stream. Key and value biases are omitted; query coordinates 1-3
        # are learned while coordinate 0 and coordinates 4-7 remain fixed.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The learned entries occupy the first four query coordinates. The
        # remaining query coordinates and every key/value bias are fixed at zero.
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
=======
        # Preserve the previously removed fourth learned coordinate while
        # ablating coordinate 0; every key/value bias remains fixed at zero.
        bias = torch.cat(
            (
                self.qkv.bias.new_zeros(1),
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model + 4),
            )
        )
>>>>>>> REPLACE