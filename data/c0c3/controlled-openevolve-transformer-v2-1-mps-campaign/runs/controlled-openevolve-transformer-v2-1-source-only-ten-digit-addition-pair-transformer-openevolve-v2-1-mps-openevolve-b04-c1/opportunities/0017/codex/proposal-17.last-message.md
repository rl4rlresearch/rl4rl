MECHANISM: Fourth-coordinate query-bias ablation

HYPOTHESIS: Learning four of eight query-bias coordinates will reduce the model from 1612 to 1611 parameters while retaining at least 99% accuracy, because the five- and six-coordinate query-bias designs both achieved 100% while preserving the necessary three-coordinate value bias.

INTENDED_EDIT: Store four learned query-bias coordinates and pad the remaining four coordinates with zeros during attention.

EVIDENCE: Successive query-bias reductions from eight to seven, six, and five learned coordinates all retained 100% accuracy, whereas reducing the three-coordinate value bias to two collapsed accuracy to 55.14%; continuing the query-bias ablation is the most informative next reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + F.pad(self.q_bias, (0, 3))
=======
        q = q + F.pad(self.q_bias, (0, 4))
>>>>>>> REPLACE