MECHANISM: Five-coordinate value-bias ablation

HYPOTHESIS: Reducing the learned value bias from four to three coordinates will lower parameters from 1616 to 1615 while retaining at least 99% accuracy, because the four-coordinate version achieved 99.98% and three learned coordinates may preserve enough of the optimization benefit lost only when all eight coordinates were removed.

INTENDED_EDIT: Store three learned value-bias coordinates and pad the remaining five coordinates with zeros during attention.

EVIDENCE: The current four-coordinate value bias reached 99.98% with 1616 parameters, while removing all eight coordinates reached 97.53%; another one-coordinate ablation is the most informative test of the parameter boundary.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 4))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 4))
=======
        v = v + F.pad(self.v_bias, (0, 5))
>>>>>>> REPLACE