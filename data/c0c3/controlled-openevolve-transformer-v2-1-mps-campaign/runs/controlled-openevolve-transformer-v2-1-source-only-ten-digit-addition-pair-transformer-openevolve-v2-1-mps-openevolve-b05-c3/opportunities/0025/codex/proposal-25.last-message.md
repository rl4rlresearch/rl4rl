MECHANISM: Alternate second hidden-bias coordinate ablation

HYPOTHESIS: Fixing hidden-bias coordinates 0 and 11 while retaining coordinates 1–10 will produce 1599 learned parameters and at least 99% accuracy, indicating that the prior 98.57% result was specific to removing coordinate 10 rather than a general two-bias capacity limit.

INTENDED_EDIT: Preserve the qualified positional gauge, four-coordinate query bias, initialization streams, optimization, and decoding, while replacing the 11 learned `fc1` bias values with ten values mapped to coordinates 1–10 and fixing coordinates 0 and 11 at zero.

EVIDENCE: Fixing coordinate 11 alone achieved 99.91% at 1600 parameters, while additionally fixing coordinate 10 narrowly missed at 98.57%; testing a nonadjacent second coordinate is the cleanest localized test of hidden-unit-specific sensitivity.

<<<<<<< SEARCH
        # Construct the baseline layer first to preserve its initialization RNG
        # stream, then fix one hidden-unit bias coordinate at zero.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
=======
        # Construct the baseline layer first to preserve its initialization RNG
        # stream, then fix the first and final hidden-unit bias coordinates at zero.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
=======
        zero = self.fc1.bias.new_zeros(1)
        bias = torch.cat((zero, self.fc1.bias, zero))
        hidden = F.linear(x, self.fc1.weight, bias)
>>>>>>> REPLACE