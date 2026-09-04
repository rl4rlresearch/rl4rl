MECHANISM: Fourth query-bias coordinate ablation

HYPOTHESIS: Learning four query-bias coordinates while retaining the qualified gauge-aware positional parameterization will produce 1601 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Fix the final four query-bias coordinates at zero while preserving all weight tensors, positional quotient updates, initialization streams, and training behavior.

EVIDENCE: Five learned query-bias coordinates achieved 99.83% accuracy at 1602 parameters, after six coordinates achieved 99.94% at 1603; this supports testing the next one-scalar localized ablation.

<<<<<<< SEARCH
        # Construct the baseline layer first to preserve the proven constructor
        # RNG stream. Key and value biases are omitted, along with one query
        # coordinate, while all weight tensors retain their baseline shapes.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Construct the baseline layer first to preserve the proven constructor
        # RNG stream. Key and value biases are omitted, and only the first four
        # query coordinates are learned; all weights retain baseline shapes.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The learned entries occupy the first seven query coordinates. The
        # final query coordinate and every key/value bias are fixed at zero.
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 1))
        )
=======
        # The learned entries occupy the first four query coordinates. The
        # remaining query coordinates and every key/value bias are fixed at zero.
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
>>>>>>> REPLACE