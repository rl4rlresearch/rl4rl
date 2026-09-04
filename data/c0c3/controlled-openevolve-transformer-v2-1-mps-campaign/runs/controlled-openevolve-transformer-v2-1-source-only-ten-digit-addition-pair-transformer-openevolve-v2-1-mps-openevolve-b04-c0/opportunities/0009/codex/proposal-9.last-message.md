MECHANISM: Single-coordinate query-bias gauge fixing

HYPOTHESIS: Fixing one query-bias coordinate at zero will reduce the model from 1628 to 1627 parameters while retaining at least 99% accuracy, because the affine first LayerNorm and learned attention projections can compensate for this single-coordinate constraint.

INTENDED_EDIT: Learn seven query-bias coordinates and append one fixed zero coordinate during attention.

EVIDENCE: The 1628-parameter bias-shared model reached 99.76%, while eight-parameter bias removals and a 17-parameter feed-forward reduction failed; this motivates the smallest possible one-parameter capacity ablation.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.v_bias = nn.Parameter(torch.zeros(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
        v = v + self.v_bias
=======
        q = q + F.pad(self.q_bias, (0, 1))
        v = v + self.v_bias
>>>>>>> REPLACE