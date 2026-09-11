MECHANISM: Six-coordinate value-bias ablation

HYPOTHESIS: Reducing the learned value bias from three coordinates to two will lower parameters from 1615 to 1614 while retaining at least 99% accuracy, because the three-coordinate design achieved 100% accuracy and two coordinates may preserve enough of the optimization benefit lost only when all eight were removed.

INTENDED_EDIT: Store two learned value-bias coordinates and pad the remaining six coordinates with zeros during attention.

EVIDENCE: The current three-coordinate value bias reached 100% accuracy with 1615 parameters, while removing all eight coordinates reached 97.53%; another one-coordinate ablation is the most informative test of the parameter boundary.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 5))
=======
        v = v + F.pad(self.v_bias, (0, 6))
>>>>>>> REPLACE