MECHANISM: Fifth-coordinate query-bias ablation

HYPOTHESIS: Learning three of eight query-bias coordinates will reduce the model from 1611 to 1610 parameters while retaining at least 99% accuracy, because the four-coordinate query-bias design achieved 99.99% while preserving the necessary three-coordinate value bias.

INTENDED_EDIT: Store three learned query-bias coordinates and pad the remaining five coordinates with zeros during attention.

EVIDENCE: Successive query-bias reductions to seven, six, five, and four learned coordinates achieved at least 99.99% accuracy, while reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the successful query-bias ablation is the most informative next reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + F.pad(self.q_bias, (0, 4))
        v = v + F.pad(self.v_bias, (0, 5))
=======
        q = q + F.pad(self.q_bias, (0, 5))
        v = v + F.pad(self.v_bias, (0, 5))
>>>>>>> REPLACE