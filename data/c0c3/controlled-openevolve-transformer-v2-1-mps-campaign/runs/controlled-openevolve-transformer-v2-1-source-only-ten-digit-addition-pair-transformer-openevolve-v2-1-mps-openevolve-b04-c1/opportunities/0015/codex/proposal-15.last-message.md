MECHANISM: Second-coordinate query-bias ablation

HYPOTHESIS: Learning six of eight query-bias coordinates will reduce the model from 1614 to 1613 parameters while retaining at least 99% accuracy, because the seven-coordinate query bias achieved 100% accuracy while preserving the verified three-coordinate value bias.

INTENDED_EDIT: Store six learned query-bias coordinates and pad the remaining two coordinates with zeros during attention.

EVIDENCE: The single-coordinate query-bias ablation reduced parameters to 1614 with 100% accuracy, whereas reducing the three-coordinate value bias caused accuracy to collapse to 55.14%; continuing the query-bias ablation is therefore the most informative next reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + F.pad(self.q_bias, (0, 1))
=======
        q = q + F.pad(self.q_bias, (0, 2))
>>>>>>> REPLACE