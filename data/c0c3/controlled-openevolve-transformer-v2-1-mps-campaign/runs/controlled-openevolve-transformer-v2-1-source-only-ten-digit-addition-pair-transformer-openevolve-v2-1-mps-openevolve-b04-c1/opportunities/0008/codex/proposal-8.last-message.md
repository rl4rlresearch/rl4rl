MECHANISM: Two-coordinate value-bias ablation

HYPOTHESIS: Removing a second value-bias coordinate will reduce parameters from 1619 to 1618 while retaining at least 99% accuracy, because the first coordinate removal preserved 99.96% accuracy and six learned coordinates still retain most of the optimization benefit lost when all eight were removed.

INTENDED_EDIT: Store six learned value-bias coordinates and pad the remaining two coordinates with zeros during attention.

EVIDENCE: The single-coordinate ablation achieved 99.96% with 1619 parameters, while removing all eight coordinates achieved only 97.53%; this motivates continuing the ablation one coordinate at a time to locate the accuracy boundary.

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