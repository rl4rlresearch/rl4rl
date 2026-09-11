MECHANISM: Third-coordinate query-bias ablation

HYPOTHESIS: Learning five of eight query-bias coordinates will reduce the model from 1613 to 1612 parameters while retaining at least 99% accuracy, because the six-coordinate query bias achieved 100% accuracy while preserving the necessary three-coordinate value bias.

INTENDED_EDIT: Store five learned query-bias coordinates and pad the remaining three coordinates with zeros during attention.

EVIDENCE: The six-coordinate query-bias design reached 100% accuracy with 1613 parameters, while reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the successful query-bias ablation is the most informative next reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 2))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 3))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + F.pad(self.q_bias, (0, 2))
        v = v + F.pad(self.v_bias, (0, 5))
=======
        q = q + F.pad(self.q_bias, (0, 3))
        v = v + F.pad(self.v_bias, (0, 5))
>>>>>>> REPLACE