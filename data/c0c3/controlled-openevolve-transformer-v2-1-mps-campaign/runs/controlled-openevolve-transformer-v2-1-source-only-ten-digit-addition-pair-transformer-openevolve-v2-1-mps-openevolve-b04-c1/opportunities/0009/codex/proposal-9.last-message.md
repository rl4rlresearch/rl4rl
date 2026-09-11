MECHANISM: Three-coordinate value-bias ablation

HYPOTHESIS: Removing one additional value-bias coordinate will reduce parameters from 1618 to 1617 while retaining at least 99% accuracy, because the two-coordinate ablation achieved 99.58% and five learned coordinates preserve most of the optimization benefit absent when all eight coordinates were removed.

INTENDED_EDIT: Store five learned value-bias coordinates and pad the remaining three coordinates with zeros during attention.

EVIDENCE: The two-coordinate ablation met the requirement at 99.58% with 1618 parameters, while removing all eight coordinates reached only 97.53%; another single-coordinate ablation is the most direct test of the capacity boundary.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        v = v + F.pad(self.v_bias, (0, 2))
=======
        v = v + F.pad(self.v_bias, (0, 3))
>>>>>>> REPLACE