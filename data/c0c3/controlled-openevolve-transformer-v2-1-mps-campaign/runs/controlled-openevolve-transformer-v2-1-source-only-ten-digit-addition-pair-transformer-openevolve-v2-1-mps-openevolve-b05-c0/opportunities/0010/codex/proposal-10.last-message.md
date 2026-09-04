MECHANISM: Second-coordinate value-bias gauge fixing

HYPOTHESIS: Fixing a second value-bias coordinate at zero will reduce the model from 1635 to 1634 parameters while retaining at least 99% accuracy, because the first one-coordinate removal achieved 99.32% and the output-projection bias preserves additive output freedom.

INTENDED_EDIT: Store six learned value-bias coordinates and append two fixed zeros before applying the bias, leaving all other architecture and training settings unchanged.

EVIDENCE: Removing one value-bias coordinate retained 99.32% accuracy at 1635 parameters, while removing the entire value bias contributed to collapse; a second single-coordinate reduction is the smallest informative continuation of that successful titration.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 1))
=======
        v = v + F.pad(self.v_bias, (0, 2))
>>>>>>> REPLACE