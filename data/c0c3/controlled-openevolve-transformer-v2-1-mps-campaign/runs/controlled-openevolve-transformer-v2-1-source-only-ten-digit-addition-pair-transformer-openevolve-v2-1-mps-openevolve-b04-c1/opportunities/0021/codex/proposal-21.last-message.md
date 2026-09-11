MECHANISM: Seventh-coordinate query-bias ablation

HYPOTHESIS: Learning one of eight query-bias coordinates will reduce the model from 1609 to 1608 parameters while retaining at least 99% accuracy, because every prior query-bias reduction through two learned coordinates retained at least 99.98% accuracy while preserving the necessary three-coordinate value bias.

INTENDED_EDIT: Store one learned query-bias coordinate and pad the remaining seven coordinates with zeros during attention.

EVIDENCE: The two-coordinate query-bias design achieved 100% accuracy with 1609 parameters, while reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the successful query-bias ablation is the most informative next reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 6))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 7))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + F.pad(self.q_bias, (0, 6))
        v = v + F.pad(self.v_bias, (0, 5))
=======
        q = q + F.pad(self.q_bias, (0, 7))
        v = v + F.pad(self.v_bias, (0, 5))
>>>>>>> REPLACE